#!/usr/bin/env python3
"""
Comprehensive orchestration tests for tennis data pipeline
Tests actual pipeline workflows and Tennis Abstract integration
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


class TestPipelineOrchestration:
    """Test actual pipeline orchestration and workflows"""

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
                ]),
                'serve_basics': pd.DataFrame([
                    {'Player_canonical': f'player_a_{i}', 'pts': 60 + i, 'aces': 5 + i, 'unret': 10 + i}
                    for i in range(50)
                ]),
                'return_outcomes': pd.DataFrame([
                    {'Player_canonical': f'player_a_{i}', 'pts': 70 + i, 'pts_won': 25 + i}
                    for i in range(50)
                ])
            },
            'women': {
                'overview': pd.DataFrame([
                    {'Player_canonical': f'player_a_{i}', 'set': 'Total', 'serve_pts': 55 + i, 'aces': 4 + i}
                    for i in range(50)
                ]),
                'serve_basics': pd.DataFrame([
                    {'Player_canonical': f'player_a_{i}', 'pts': 55 + i, 'aces': 4 + i, 'unret': 8 + i}
                    for i in range(50)
                ]),
                'return_outcomes': pd.DataFrame([
                    {'Player_canonical': f'player_a_{i}', 'pts': 65 + i, 'pts_won': 22 + i}
                    for i in range(50)
                ])
            }
        }

    def create_weighted_defaults(self):
        """Create comprehensive weighted defaults"""
        return {
            'men': {
                'serve_pts': 65.0, 'aces': 6.2, 'return_pts_won': 25.8,
                'first_serve_pct': 0.62, 'winners_total': 28.5, 'unforced_errors': 26.3
            },
            'women': {
                'serve_pts': 60.0, 'aces': 4.8, 'return_pts_won': 23.2,
                'first_serve_pct': 0.59, 'winners_total': 24.7, 'unforced_errors': 28.1
            }
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
            },
            {
                'composite_id': '20250715-wimbledon-jannik_sinner-carlos_alcaraz',
                'Player_canonical': 'carlos_alcaraz',
                'data_type': 'serve',
                'stat_name': 'points_won_pct',
                'stat_value': 68.2,
                'Date': '20250715',
                'tournament': 'Wimbledon',
                'player1': 'Jannik Sinner',
                'player2': 'Carlos Alcaraz'
            },
            {
                'composite_id': '20250715-wimbledon-jannik_sinner-carlos_alcaraz',
                'Player_canonical': 'jannik_sinner',
                'data_type': 'keypoints',
                'stat_name': 'serve_won_pct',
                'stat_value': 82.1,
                'Date': '20250715',
                'tournament': 'Wimbledon',
                'player1': 'Jannik Sinner',
                'player2': 'Carlos Alcaraz'
            }
        ]

    # TEST 1: COLD START ORCHESTRATION (EMPTY CACHE)
    @patch('tennis_updated.load_all_tennis_data')
    @patch('tennis_updated.load_jeff_comprehensive_data')
    @patch('tennis_updated.calculate_comprehensive_weighted_defaults')
    @patch('tennis_updated.api_call')
    @patch('tennis_updated.AutomatedTennisAbstractScraper')
    def test_1_cold_start_orchestration(self, mock_scraper, mock_api, mock_defaults, mock_jeff, mock_tennis):
        """Test complete cold start pipeline orchestration"""
        print("\n=== TESTING COLD START ORCHESTRATION ===")

        # Setup comprehensive mocks
        mock_tennis.return_value = self.create_realistic_historical_data(500)
        mock_jeff.return_value = self.create_comprehensive_jeff_data()
        mock_defaults.return_value = self.create_weighted_defaults()
        mock_api.return_value = []  # No API data initially

        # Mock Tennis Abstract scraper
        mock_scraper_instance = MagicMock()
        mock_scraper_instance.automated_scraping_session.return_value = self.create_tennis_abstract_scraped_data()
        mock_scraper.return_value = mock_scraper_instance

        # Verify empty cache
        assert not os.path.exists(self.hd_path), "Cache should be empty initially"

        # Execute cold start orchestration
        hist, jeff_data, defaults = load_from_cache_with_scraping()

        # Verify orchestration results
        assert hist is not None, "Should generate historical data"
        assert len(hist) >= 500, "Should contain base tennis data"
        assert jeff_data is not None, "Should have Jeff data"
        assert defaults is not None, "Should have weighted defaults"

        # Verify cache files created
        assert os.path.exists(self.hd_path), "Historical data cache should be created"
        assert os.path.exists(self.jeff_path), "Jeff data cache should be created"
        assert os.path.exists(self.def_path), "Defaults cache should be created"

        # Verify Tennis Abstract integration was called
        mock_scraper.assert_called_once()
        mock_scraper_instance.automated_scraping_session.assert_called_once()

        print("✓ Cold start orchestration completed successfully")

    # TEST 2: INCREMENTAL UPDATE ORCHESTRATION
    @patch('tennis_updated.AutomatedTennisAbstractScraper')
    @patch('tennis_updated.api_call')
    @patch('tennis_updated.get_player_rankings')
    @patch('tennis_updated.get_tournaments_metadata')
    def test_2_incremental_update_orchestration(self, mock_tournaments, mock_rankings, mock_api, mock_scraper):
        """Test incremental update orchestration with existing cache"""
        print("\n=== TESTING INCREMENTAL UPDATE ORCHESTRATION ===")

        # Setup existing cache
        existing_data = self.create_realistic_historical_data(300)
        existing_data['date'] = existing_data['date'].apply(lambda x: x - timedelta(days=5))  # Older data
        existing_jeff = self.create_comprehensive_jeff_data()
        existing_defaults = self.create_weighted_defaults()

        save_to_cache(existing_data, existing_jeff, existing_defaults)
        original_count = len(existing_data)

        # Mock new API data
        mock_api.return_value = [{
            'event_status': 'Finished',
            'first_player_key': '1001',
            'second_player_key': '1002',
            'event_first_player': 'New Player A',
            'event_second_player': 'New Player B',
            'event_winner': 'First Player',
            'tournament_name': 'New Tournament',
            'event_type_type': 'ATP Singles'
        }]

        mock_rankings.return_value = {'1001': 10, '1002': 15}
        mock_tournaments.return_value = {}

        # Mock Tennis Abstract with new data
        mock_scraper_instance = MagicMock()
        mock_scraper_instance.automated_scraping_session.return_value = self.create_tennis_abstract_scraped_data()
        mock_scraper.return_value = mock_scraper_instance

        # Execute incremental update
        with patch('tennis_updated.get_event_types', return_value={}), \
                patch('tennis_updated.get_h2h_data',
                      return_value={'h2h_matches': 0, 'p1_wins': 0, 'p2_wins': 0, 'p1_win_pct': 0.5}), \
                patch('tennis_updated.get_match_odds', return_value=(None, None)), \
                patch('tennis_updated.get_player_profile', return_value={}):
            hist, _, _ = load_from_cache_with_scraping()

        # Verify incremental updates
        assert len(hist) >= original_count, "Should maintain or increase data count"

        # Check for Tennis Abstract enhancements
        ta_columns = [col for col in hist.columns if col.startswith(('winner_ta_', 'loser_ta_'))]
        if ta_columns:
            ta_enhanced_matches = hist[hist[ta_columns].notna().any(axis=1)]
            assert len(ta_enhanced_matches) > 0, "Should have TA-enhanced matches"

        print("✓ Incremental update orchestration completed successfully")

    # TEST 3: TENNIS ABSTRACT BACKFILL ORCHESTRATION
    def test_3_tennis_abstract_backfill_orchestration(self):
        """Test Tennis Abstract backfill logic and integration"""
        print("\n=== TESTING TENNIS ABSTRACT BACKFILL ORCHESTRATION ===")

        # Create historical data without TA features
        hist_data = self.create_realistic_historical_data(200)
        hist_data['ta_enhanced'] = False

        # Create Tennis Abstract scraped records
        scraped_records = self.create_tennis_abstract_scraped_data()

        # Test Tennis Abstract integration workflow
        enhanced_data = integrate_scraped_data_hybrid(hist_data, scraped_records)

        # Verify enhancement
        assert len(enhanced_data) >= len(hist_data), "Should maintain or increase data count"

        # Check for TA feature columns
        ta_columns = [col for col in enhanced_data.columns if col.startswith(('winner_ta_', 'loser_ta_'))]
        assert len(ta_columns) > 0, "Should have Tennis Abstract feature columns"

        # Test scraped data processing
        processed_matches = process_tennis_abstract_scraped_data(scraped_records)
        assert len(processed_matches) > 0, "Should process scraped records"

        # Verify composite_id based matching
        wimbledon_match = '20250715-wimbledon-jannik_sinner-carlos_alcaraz'
        if wimbledon_match in processed_matches:
            match_data = processed_matches[wimbledon_match]
            assert 'jannik_sinner' in match_data or 'carlos_alcaraz' in match_data, "Should have player data"

        print("✓ Tennis Abstract backfill orchestration completed successfully")

    # TEST 4: TENNIS ABSTRACT SCRAPER ORCHESTRATION
    @patch('requests.get')
    def test_4_tennis_abstract_scraper_orchestration(self, mock_get):
        """Test actual Tennis Abstract scraper orchestration"""
        print("\n=== TESTING TENNIS ABSTRACT SCRAPER ORCHESTRATION ===")

        # Mock HTTP responses
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

        # Test scraper instantiation and method calls
        scraper = TennisAbstractScraper()

        test_url = "https://www.tennisabstract.com/charting/20250715-M-Wimbledon-F-Jannik_Sinner-Carlos_Alcaraz.html"

        # Test URL parsing
        match_meta = scraper._parse_match_url(test_url)
        assert 'Date' in match_meta, "Should parse match metadata"
        assert match_meta['Date'] == '20250715', "Should extract correct date"

        # Test comprehensive scraping
        scraped_data = scraper.scrape_comprehensive_match_data(test_url)

        # Verify scraping results
        assert isinstance(scraped_data, list), "Should return list of records"
        # Note: Detailed assertions depend on mock HTML structure

        print("✓ Tennis Abstract scraper orchestration completed successfully")

    # TEST 5: FULL PIPELINE SCRIPT ORCHESTRATION
    def test_5_full_pipeline_script_orchestration(self):
        """Test that full_data_pipeline.py script can be invoked"""
        print("\n=== TESTING FULL PIPELINE SCRIPT ORCHESTRATION ===")

        # Create a minimal full_data_pipeline.py for testing
        pipeline_script = """
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tennis_updated import load_from_cache_with_scraping, save_to_cache

try:
    print("Starting pipeline orchestration...")
    hist, jeff_data, defaults = load_from_cache_with_scraping()

    if hist is not None:
        print(f"Pipeline completed: {len(hist)} matches processed")
        print("SUCCESS")
    else:
        print("Pipeline failed: No data generated")
        print("FAILED")

except Exception as e:
    print(f"Pipeline error: {e}")
    print("ERROR")
"""

        # Write test script
        script_path = os.path.join(self.temp_cache_dir, "test_pipeline.py")
        with open(script_path, 'w') as f:
            f.write(pipeline_script)

        # Mock external dependencies for the subprocess call
        with patch('tennis_updated.load_all_tennis_data') as mock_tennis, \
                patch('tennis_updated.load_jeff_comprehensive_data') as mock_jeff, \
                patch('tennis_updated.api_call') as mock_api:

            mock_tennis.return_value = self.create_realistic_historical_data(50)
            mock_jeff.return_value = self.create_comprehensive_jeff_data()
            mock_api.return_value = []

            # Execute script as subprocess (more realistic)
            try:
                result = subprocess.run([
                    'python', script_path
                ], capture_output=True, text=True, timeout=30,
                    env={**os.environ, 'PYTHONPATH': os.path.dirname(os.path.dirname(__file__))})

                # Check results
                assert result.returncode == 0, f"Script failed with return code {result.returncode}"
                assert "SUCCESS" in result.stdout or "completed" in result.stdout, "Script should indicate success"

            except subprocess.TimeoutExpired:
                pytest.fail("Pipeline script timed out")
            except Exception as e:
                pytest.fail(f"Pipeline script execution failed: {e}")

        print("✓ Full pipeline script orchestration completed successfully")

    # TEST 6: DATA FLOW INTEGRITY ORCHESTRATION
    def test_6_data_flow_integrity_orchestration(self):
        """Test data integrity through the complete orchestration flow"""
        print("\n=== TESTING DATA FLOW INTEGRITY ORCHESTRATION ===")

        # Create initial dataset with known characteristics
        initial_data = self.create_realistic_historical_data(100)
        initial_composite_ids = set(initial_data['composite_id'])

        # Save initial state
        save_to_cache(initial_data, self.create_comprehensive_jeff_data(), self.create_weighted_defaults())

        # Mock external calls for controlled testing
        with patch('tennis_updated.api_call', return_value=[]), \
                patch('tennis_updated.AutomatedTennisAbstractScraper') as mock_scraper:
            # Mock scraper to return controlled TA data
            mock_scraper_instance = MagicMock()
            mock_scraper_instance.automated_scraping_session.return_value = []  # No new TA data
            mock_scraper.return_value = mock_scraper_instance

            # Execute full orchestration
            hist, jeff_data, defaults = load_from_cache_with_scraping()

        # Verify data integrity
        final_composite_ids = set(hist['composite_id'])

        # All original IDs should still exist
        missing_ids = initial_composite_ids - final_composite_ids
        assert len(missing_ids) == 0, f"Lost composite_ids during orchestration: {missing_ids}"

        # Check data quality metrics
        assert hist['composite_id'].nunique() == len(hist), "Should have unique composite_ids"
        assert hist['winner_canonical'].notna().all(), "Should have winner_canonical for all matches"
        assert hist['loser_canonical'].notna().all(), "Should have loser_canonical for all matches"

        # Verify source ranking integrity
        if 'source_rank' in hist.columns:
            valid_source_ranks = hist['source_rank'].isin([1, 2, 3]).all()
            assert valid_source_ranks, "Should have valid source ranks (1, 2, or 3)"

        print("✓ Data flow integrity orchestration completed successfully")

    # TEST 7: PERFORMANCE ORCHESTRATION
    def test_7_performance_orchestration(self):
        """Test performance characteristics of orchestration"""
        print("\n=== TESTING PERFORMANCE ORCHESTRATION ===")

        # Test with different data sizes
        small_data = self.create_realistic_historical_data(50)
        medium_data = self.create_realistic_historical_data(500)

        # Mock external dependencies
        with patch('tennis_updated.api_call', return_value=[]), \
                patch('tennis_updated.AutomatedTennisAbstractScraper') as mock_scraper:
            mock_scraper_instance = MagicMock()
            mock_scraper_instance.automated_scraping_session.return_value = []
            mock_scraper.return_value = mock_scraper_instance

            # Time small dataset processing
            save_to_cache(small_data, self.create_comprehensive_jeff_data(), self.create_weighted_defaults())

            start_time = time.time()
            hist_small, _, _ = load_from_cache_with_scraping()
            small_duration = time.time() - start_time

            # Time medium dataset processing
            save_to_cache(medium_data, self.create_comprehensive_jeff_data(), self.create_weighted_defaults())

            start_time = time.time()
            hist_medium, _, _ = load_from_cache_with_scraping()
            medium_duration = time.time() - start_time

        # Performance assertions
        assert small_duration < 10.0, "Small dataset processing should be fast"
        assert medium_duration < 30.0, "Medium dataset processing should be reasonable"

        # Scaling should be reasonable (not quadratic)
        scaling_factor = medium_duration / small_duration if small_duration > 0 else 1
        assert scaling_factor < 20, "Performance should scale reasonably with data size"

        print(f"✓ Performance orchestration: small={small_duration:.2f}s, medium={medium_duration:.2f}s")

    # TEST 8: ERROR RECOVERY ORCHESTRATION
    def test_8_error_recovery_orchestration(self):
        """Test error handling and recovery in orchestration"""
        print("\n=== TESTING ERROR RECOVERY ORCHESTRATION ===")

        # Setup valid initial state
        valid_data = self.create_realistic_historical_data(50)
        save_to_cache(valid_data, self.create_comprehensive_jeff_data(), self.create_weighted_defaults())
        initial_checksum = self._get_file_checksum(self.hd_path)

        # Test API failure recovery
        with patch('tennis_updated.api_call', side_effect=Exception("API failed")), \
                patch('tennis_updated.AutomatedTennisAbstractScraper') as mock_scraper:

            mock_scraper_instance = MagicMock()
            mock_scraper_instance.automated_scraping_session.return_value = []
            mock_scraper.return_value = mock_scraper_instance

            # Should handle API failure gracefully
            try:
                hist, _, _ = load_from_cache_with_scraping()
                # Should still return cached data
                assert hist is not None, "Should return cached data on API failure"
                assert len(hist) >= 50, "Should maintain cached data count"
            except Exception as e:
                pytest.fail(f"Orchestration should handle API failures gracefully: {e}")

        # Verify cache integrity after failure
        post_error_checksum = self._get_file_checksum(self.hd_path)
        # Cache should either be unchanged or updated safely
        assert post_error_checksum is not None, "Cache should remain valid after error"

        print("✓ Error recovery orchestration completed successfully")

    def _get_file_checksum(self, filepath):
        """Helper to get file checksum"""
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'rb') as f:
            return hash(f.read())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--capture=no"])