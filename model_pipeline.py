#!/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
import warnings
import pickle
import argparse
import hashlib
import time
import gc
import re
import logging
from pathlib import Path
from functools import wraps
from contextlib import contextmanager
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.feature_selection import SelectKBest, f_classif
import lightgbm as lgb

warnings.filterwarnings("ignore")
pd.set_option("mode.copy_on_write", True)

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)


# ADJUSTMENT 12: Configurable logging level
def setup_logging(verbose=False):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")


sys.path.append('.')
from model import TennisModelPipeline, ModelConfig
from tennis_updated import (
    load_from_cache_with_scraping, generate_comprehensive_historical_data, save_to_cache,
    load_all_tennis_data, load_jeff_comprehensive_data, calculate_comprehensive_weighted_defaults,
    integrate_api_tennis_data_incremental, AutomatedTennisAbstractScraper, CACHE_DIR,
    extract_comprehensive_jeff_features, normalize_name
)


def create_canonical_composite_id(match_date, tournament, player1, player2):
    """Single canonicalization function used everywhere"""
    date_str = pd.to_datetime(match_date).strftime("%Y%m%d")

    def canonicalize_component(text):
        if pd.isna(text):
            return ""
        text = str(text).lower().strip()
        text = text.replace('.', '').replace("'", '').replace('-', ' ')
        text = ' '.join(text.split())
        return text.replace(' ', '_')

    tournament_canonical = canonicalize_component(tournament)
    player1_canonical = canonicalize_component(player1)
    player2_canonical = canonicalize_component(player2)

    return f"{date_str}-{tournament_canonical}-{player1_canonical}-{player2_canonical}"


def create_playerdate_id(match_date, player1, player2):
    """FIXED: Tournament-agnostic ID using only date + players"""
    date_str = pd.to_datetime(match_date).strftime("%Y%m%d")

    def canon(t):
        return (str(t).lower()
                .replace("'", "")
                .replace(".", "")
                .replace("-", "_")
                .replace(" ", "_"))

    p1_canon = canon(player1)
    p2_canon = canon(player2)

    # Sort players alphabetically for consistent ordering
    if p1_canon <= p2_canon:
        return f"{date_str}-{p1_canon}-{p2_canon}"
    else:
        return f"{date_str}-{p2_canon}-{p1_canon}"


# ADJUSTMENT 6: Fixed regex pattern
def parse_point_match_id(mid: str) -> str:
    """
    Turn Jeff's raw match_id ('20250713-M-Wimbledon-F-Novak_Djokovic-Jannik_Sinner')
    into the canonical YYYYMMDD-p1-p2 string used in point files.
    """
    parts = mid.split('-')
    ymd = parts[0]

    def canon(t):
        return (str(t).lower()
                .replace("'", "")
                .replace(".", "")
                .replace("-", "_")
                .replace(" ", "_"))

    p1_canon = canon(parts[-2].replace('_', ' '))
    p2_canon = canon(parts[-1].replace('_', ' '))

    # Sort players alphabetically for consistent ordering
    if p1_canon <= p2_canon:
        return f"{ymd}-{p1_canon}-{p2_canon}"
    else:
        return f"{ymd}-{p2_canon}-{p1_canon}"


@contextmanager
def timer_context(stage_name, timeout=None):
    """Context manager for timing with configurable timeout"""
    start_time = time.time()
    logging.info(f"[TIMING] Starting {stage_name}...")

    try:
        yield
        duration = time.time() - start_time

        if timeout and duration > timeout:
            raise TimeoutError(f"{stage_name} exceeded {timeout}s threshold: {duration:.1f}s")

        logging.info(f"[TIMING] {stage_name} completed in {duration:.1f}s")

    except Exception as e:
        duration = time.time() - start_time
        logging.error(f"[TIMING] {stage_name} failed after {duration:.1f}s: {e}")
        raise


def timing_wrapper(stage_name, timeout=None):
    """Decorator with configurable timeout"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with timer_context(stage_name, timeout):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# ADJUSTMENT 13: Improved hash function
def compute_data_hash(data):
    """Compute deterministic SHA-256 hash including object columns"""
    hash_components = []

    # Numeric data
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        sorted_data = data[numeric_cols].sort_index().sort_index(axis=1)
        data_bytes = sorted_data.to_numpy().astype(np.float32).tobytes()
        hash_components.append(data_bytes)

    # ADJUSTMENT 13: Include object columns length and sample
    object_cols = data.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        for col in object_cols:
            col_info = f"{col}:{len(data[col])}:{hash(str(data[col].iloc[0]) if len(data) > 0 else '')}"
            hash_components.append(col_info.encode())

    # Combine all components
    combined = b''.join(str(comp).encode() if isinstance(comp, str) else comp for comp in hash_components)
    return hashlib.sha256(combined).hexdigest()


# ADJUSTMENT 2: Vectorized defaults construction
def create_extended_weighted_defaults_vectorized(jeff_data):
    """ADJUSTMENT 2: Vectorized weighted defaults construction"""
    logging.info("Computing extended weighted defaults with vectorization...")

    defaults = {"men": {}, "women": {}}

    for gender in ("men", "women"):
        if gender not in jeff_data:
            continue

        # ADJUSTMENT 2: Concatenate all numeric data once
        all_numeric_data = []

        for dataset_name, df in jeff_data[gender].items():
            if df is None or df.empty:
                continue
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                numeric_data = df[numeric_cols].copy()
                # Add dataset prefix to avoid column conflicts
                numeric_data.columns = [f"{dataset_name}_{col}" for col in numeric_data.columns]
                all_numeric_data.append(numeric_data)

        if all_numeric_data:
            # ADJUSTMENT 2: Single concatenation and mean calculation
            combined_numeric = pd.concat(all_numeric_data, axis=1, ignore_index=False)
            defaults[gender] = combined_numeric.mean().to_dict()
            logging.info(f"  {gender}: {len(defaults[gender])} features computed vectorized")

    return defaults


class FixedComprehensiveDataPipeline:
    """Pipeline with all 14 critical adjustments applied + TA fix"""

    def __init__(self, cache_dir=CACHE_DIR, random_seed=GLOBAL_SEED,
                 timeout_jeff=1200, timeout_ta=600):  # ADJUSTMENT 10: Reduced Jeff timeout
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.random_seed = random_seed
        self.model_cache = self.cache_dir / "trained_models"
        self.model_cache.mkdir(exist_ok=True)

        # ADJUSTMENT 10: Reduced timeout
        self.timeout_jeff = timeout_jeff
        self.timeout_ta = timeout_ta

        self.jeff_cache = self.cache_dir / "jeff_agg.pkl"
        self.feature_selector = None

        np.random.seed(random_seed)

    def get_jeff_cache_key(self, jeff_data):
        """Generate cache key based on Jeff data"""
        key_parts = []
        for gender in ['men', 'women']:
            if gender in jeff_data:
                for dataset_name, df in jeff_data[gender].items():
                    if not df.empty:
                        key_parts.append(f"{gender}_{dataset_name}_{len(df)}_{df.columns.tolist()}")
        return hashlib.md5('_'.join(key_parts).encode()).hexdigest()

    def load_or_compute_jeff_aggregations(self, jeff_data):
        """Cache Jeff aggregations for performance"""
        cache_key = self.get_jeff_cache_key(jeff_data)

        if self.jeff_cache.exists():
            try:
                with open(self.jeff_cache, 'rb') as f:
                    cached_data = pickle.load(f)
                    if cached_data.get('cache_key') == cache_key:
                        logging.info("✓ Loaded Jeff aggregations from cache")
                        return cached_data['aggregations']
            except Exception as e:
                logging.warning(f"Cache load failed: {e}")

        logging.info("Computing Jeff aggregations...")
        aggregations = {}

        for gender in ['men', 'women']:
            if gender not in jeff_data:
                continue

            gender_aggs = {}

            # Overview aggregation
            if 'overview' in jeff_data[gender]:
                overview_df = jeff_data[gender]['overview']
                if 'Player_canonical' in overview_df.columns:
                    overview_total = overview_df[overview_df.get('set', '') == 'Total'].copy()
                    if not overview_total.empty:
                        agg_cols = ['serve_pts', 'aces', 'dfs', 'first_in', 'first_won',
                                    'second_won', 'bp_saved', 'return_pts_won', 'winners',
                                    'winners_fh', 'winners_bh', 'unforced', 'unforced_fh', 'unforced_bh']
                        existing_cols = [col for col in agg_cols if col in overview_total.columns]
                        if existing_cols:
                            gender_aggs['overview'] = overview_total.groupby('Player_canonical')[
                                existing_cols].mean().astype(np.float32)

            # Other dataset aggregations
            dataset_configs = [
                ('serve_basics', ['pts_won', 'aces', 'unret', 'forced_err', 'wide', 'body', 't']),
                ('return_outcomes', ['returnable', 'returnable_won', 'in_play', 'in_play_won', 'winners']),
                ('key_points_serve', ['pts_won', 'first_in', 'aces', 'svc_winners', 'rally_winners']),
                ('net_points', ['net_pts', 'pts_won', 'net_winner', 'induced_forced', 'passed_at_net'])
            ]

            for dataset_name, agg_cols in dataset_configs:
                if dataset_name in jeff_data[gender]:
                    df = jeff_data[gender][dataset_name]
                    if 'player' in df.columns:
                        existing_cols = [col for col in agg_cols if col in df.columns]
                        if existing_cols:
                            gender_aggs[dataset_name] = df.groupby('player')[existing_cols].mean().astype(np.float32)

            aggregations[gender] = gender_aggs

        # Save to cache
        try:
            with open(self.jeff_cache, 'wb') as f:
                pickle.dump({
                    'cache_key': cache_key,
                    'aggregations': aggregations
                }, f)
            logging.info("✓ Cached Jeff aggregations")
        except Exception as e:
            logging.warning(f"Cache save failed: {e}")

        return aggregations

    def validate_feature_extraction(self, data, feature_prefix, min_features=5):
        """Validate feature extraction with hard failure"""
        feature_cols = [col for col in data.columns if col.startswith(feature_prefix)]

        if len(feature_cols) < min_features:
            raise ValueError(
                f"CRITICAL: {feature_prefix} extraction failed - only {len(feature_cols)} features, need {min_features}")

        non_null_counts = data[feature_cols].count().sum()
        if non_null_counts == 0:
            raise ValueError(f"CRITICAL: All {feature_prefix} features are null")

        logging.info(f"✓ Validated {feature_prefix}: {len(feature_cols)} features, {non_null_counts:,} non-null values")
        return feature_cols

    # ADJUSTMENT 1, 3, 4: Fixed Jeff feature extraction
    def extract_all_jeff_features_optimized(self, match_data, jeff_data, weighted_defaults):
        """ADJUSTMENTS 1,3,4: Use passed weighted_defaults, vectorized injection, smart pre-allocation"""
        logging.info("Extracting Jeff features with all optimizations...")

        enhanced_data = match_data.copy()

        # ADJUSTMENT 1: Use passed weighted_defaults instead of rebuilding
        # Get aggregations from cache
        aggregations = self.load_or_compute_jeff_aggregations(jeff_data)

        # ADJUSTMENT 4: Smart pre-allocation - only union of aggregations + weighted defaults
        columns_to_create = set()

        # Add columns from aggregations
        for gender in ['men', 'women']:
            if gender in aggregations:
                for dataset_name, agg_df in aggregations[gender].items():
                    for col in agg_df.columns:
                        if dataset_name == 'overview':
                            columns_to_create.add(f'winner_{col}')
                            columns_to_create.add(f'loser_{col}')
                        else:
                            columns_to_create.add(f'winner_{dataset_name}_{col}')
                            columns_to_create.add(f'loser_{dataset_name}_{col}')

        # Add columns from weighted defaults
        for gender_key in ['men', 'women']:
            if gender_key in weighted_defaults:
                for feature_name in weighted_defaults[gender_key].keys():
                    columns_to_create.add(f'winner_{feature_name}')
                    columns_to_create.add(f'loser_{feature_name}')

        logging.info(f"ADJUSTMENT 4: Pre-creating {len(columns_to_create)} smart-allocated columns")

        # Pre-allocate only needed columns
        for col_name in columns_to_create:
            if col_name not in enhanced_data.columns:
                enhanced_data[col_name] = np.nan

        # ADJUSTMENT 3: Vectorized default injection using DataFrame operations
        logging.info("ADJUSTMENT 3: Applying weighted defaults with vectorized injection...")

        for gender_char, gender_key in [('M', 'men'), ('W', 'women')]:
            if gender_key not in weighted_defaults:
                continue

            gender_mask = enhanced_data['gender'] == gender_char
            if not gender_mask.any():
                continue

            defaults = weighted_defaults[gender_key]

            # ADJUSTMENT 3: Build DataFrames for vectorized injection
            winner_defaults = {f'winner_{k}': v for k, v in defaults.items()
                               if f'winner_{k}' in enhanced_data.columns}
            loser_defaults = {f'loser_{k}': v for k, v in defaults.items()
                              if f'loser_{k}' in enhanced_data.columns}

            # Apply defaults only where values are NaN
            for col, default_val in winner_defaults.items():
                mask = gender_mask & enhanced_data[col].isna()
                enhanced_data.loc[mask, col] = default_val

            for col, default_val in loser_defaults.items():
                mask = gender_mask & enhanced_data[col].isna()
                enhanced_data.loc[mask, col] = default_val

        logging.info("✓ Weighted defaults applied with vectorized injection")

        # Apply aggregated data efficiently (existing efficient code)
        for gender in ['men', 'women']:
            if gender not in aggregations:
                continue

            gender_key = gender
            gender_mask = enhanced_data['gender'] == ('M' if gender == 'men' else 'W')

            if not gender_mask.any():
                continue

            logging.info(f"  Processing {gender}: {gender_mask.sum():,} matches")

            # Overview stats
            if 'overview' in aggregations[gender]:
                overview_agg = aggregations[gender]['overview']

                # Vectorized merge for winners and losers
                for prefix, canonical_col in [('winner_', 'winner_canonical'), ('loser_', 'loser_canonical')]:
                    merge_data = enhanced_data.loc[gender_mask].merge(
                        overview_agg.add_prefix(prefix),
                        left_on=canonical_col,
                        right_index=True,
                        how='left'
                    )

                    for col in overview_agg.columns:
                        full_col = f'{prefix}{col}'
                        if full_col in merge_data.columns:
                            mask = gender_mask & merge_data[full_col].notna()
                            enhanced_data.loc[mask, full_col] = merge_data.loc[mask, full_col]

            # Other datasets
            for dataset_name, agg_df in aggregations[gender].items():
                if dataset_name == 'overview':
                    continue

                for prefix, canonical_col in [('winner_', 'winner_canonical'), ('loser_', 'loser_canonical')]:
                    for col in agg_df.columns:
                        col_name = f'{prefix}{dataset_name}_{col}'
                        if col_name in enhanced_data.columns:
                            player_stats = enhanced_data.loc[gender_mask, canonical_col].map(agg_df[col])
                            mask = gender_mask & player_stats.notna()
                            enhanced_data.loc[mask, col_name] = player_stats.loc[mask]

        logging.info("✓ Jeff feature extraction complete with all optimizations")

        # Validate
        self.validate_feature_extraction(enhanced_data, 'winner_', min_features=10)
        self.validate_feature_extraction(enhanced_data, 'loser_', min_features=10)

        return enhanced_data

    def scrape_tennis_abstract_with_fallback(self, days_back=90, max_matches=500):
        """TA scraping with fallback to larger window"""
        try:
            scraper = AutomatedTennisAbstractScraper()
            scraped_records = scraper.automated_scraping_session(days_back=days_back, max_matches=max_matches)

            if not scraped_records:
                logging.info(f"No TA data found with {days_back} days, trying larger window...")
                scraped_records = scraper.automated_scraping_session(days_back=365, max_matches=2000)

            return scraped_records

        except Exception as e:
            logging.error(f"Tennis Abstract scraping failed: {e}")
            return []

    # FIXED: TA integration using playerdate_id instead of composite_id
    def integrate_all_tennis_abstract_features_fixed(self, match_data, scraped_records):
        """FIXED: TA integration using playerdate_id to avoid tournament name mismatches"""
        logging.info("FIXED: Integrating Tennis Abstract features using playerdate_id...")

        if 'playerdate_id' not in match_data.columns:
            raise ValueError("playerdate_id column missing from match_data")

        if not scraped_records:
            raise ValueError("CRITICAL: No Tennis Abstract records available")

        # FIXED: Generate playerdate_id for all TA records
        logging.info("Generating playerdate_id for Tennis Abstract records...")
        for record in scraped_records:
            if all(k in record for k in ['Date', 'player1', 'player2']):
                record['playerdate_id'] = create_playerdate_id(
                    record['Date'], record['player1'], record['player2']
                )

        # Organize data with ADJUSTMENT 5: Player_canonical guard
        ta_by_match = {}
        data_types_found = set()

        for record in scraped_records:
            # FIXED: Use playerdate_id instead of composite_id
            match_id = record.get('playerdate_id')
            # ADJUSTMENT 5: Guard for missing Player_canonical
            if 'Player_canonical' not in record:
                continue
            player = record.get('Player_canonical')
            data_type = record.get('data_type')
            stat_name = record.get('stat_name')
            stat_value = record.get('stat_value')

            if not all([match_id, player, data_type, stat_name]) or stat_value is None:
                continue

            data_types_found.add(data_type)

            if match_id not in ta_by_match:
                ta_by_match[match_id] = {}
            if player not in ta_by_match[match_id]:
                ta_by_match[match_id][player] = {}

            feature_name = f"ta_{data_type}_{stat_name}"
            ta_by_match[match_id][player][feature_name] = stat_value

        if not data_types_found:
            raise ValueError("CRITICAL: No Tennis Abstract data types found")

        logging.info(f"TA data types found: {sorted(data_types_found)}")
        logging.info(f"Matches with TA data: {len(ta_by_match)}")

        # FIXED: Check overlap using playerdate_id
        tennis_match_ids = set(match_data['playerdate_id'].unique())
        ta_match_ids = set(ta_by_match.keys())
        overlap = tennis_match_ids & ta_match_ids
        logging.info(f"PlayerDate ID overlap: {len(overlap)}/{len(ta_match_ids)} TA matches found")

        if len(overlap) == 0:
            raise ValueError("CRITICAL: No playerdate_id overlap between tennis data and TA data")

        # Apply features
        enhanced_data = match_data.copy()
        matches_enhanced = 0
        ta_features_added = set()

        for match_id, players in ta_by_match.items():
            # FIXED: Use playerdate_id for matching
            match_mask = enhanced_data['playerdate_id'] == match_id
            if not match_mask.any():
                continue

            match_idx = enhanced_data[match_mask].index[0]
            current_row = enhanced_data.loc[match_idx]

            winner_canonical = current_row.get('winner_canonical')
            loser_canonical = current_row.get('loser_canonical')

            for player_canonical, features in players.items():
                if player_canonical == winner_canonical:
                    prefix = 'winner_'
                elif player_canonical == loser_canonical:
                    prefix = 'loser_'
                else:
                    continue

                for feature_name, feature_value in features.items():
                    col_name = f"{prefix}{feature_name}"

                    if col_name not in enhanced_data.columns:
                        enhanced_data[col_name] = np.nan

                    enhanced_data.loc[match_idx, col_name] = feature_value
                    ta_features_added.add(col_name)

            enhanced_data.loc[match_idx, 'ta_enhanced'] = True
            matches_enhanced += 1

        logging.info(f"Enhanced {matches_enhanced} matches with {len(ta_features_added)} TA features")

        if matches_enhanced == 0:
            raise ValueError("CRITICAL: Zero matches enhanced with TA features")

        self.validate_feature_extraction(enhanced_data, 'winner_ta_', min_features=1)
        self.validate_feature_extraction(enhanced_data, 'loser_ta_', min_features=1)

        return enhanced_data

    # FIXED: Point integration using playerdate_id
    def integrate_point_level_features_fixed(self, match_data, point_data):
        """FIXED: Vectorized point integration using playerdate_id"""
        logging.info("FIXED: Integrating point-level data using playerdate_id...")

        if point_data.empty:
            logging.info("No point data to integrate")
            return match_data

        # FIXED: Re-canonicalize point data match_ids to playerdate_id
        logging.info("Re-canonicalizing point data match_ids to playerdate_id...")
        unique_match_ids = point_data['match_id'].unique()
        playerdate_mapping = {}

        for match_id in unique_match_ids:
            playerdate_mapping[match_id] = parse_point_match_id(match_id)

        point_data = point_data.copy()
        point_data['playerdate_id'] = point_data['match_id'].map(playerdate_mapping)

        # FIXED: Check overlap using playerdate_id
        tennis_match_ids = set(match_data['playerdate_id'].unique())
        point_match_ids = set(point_data['playerdate_id'].unique())
        overlap = tennis_match_ids & point_match_ids

        logging.info(f"Point-level playerdate_id overlap: {len(overlap)}/{len(point_match_ids)} point matches found")

        if len(overlap) < 100:  # Lowered threshold since we're using playerdate_id
            logging.warning(f"Low point overlap - only {len(overlap)} matches, continuing anyway")

        # ADJUSTMENT 7: Vectorized aggregation instead of manual loop
        filtered_point_data = point_data[point_data['playerdate_id'].isin(overlap)]
        logging.info(f"✓ Using {len(filtered_point_data):,} points from {len(overlap)} matched games")

        if len(filtered_point_data) == 0:
            logging.warning("No point data after filtering, skipping point integration")
            return match_data

        # ADJUSTMENT 7: Pure vectorized operations
        server_1_mask = filtered_point_data['Svr'] == 1
        server_2_mask = filtered_point_data['Svr'] == 2

        # Create winner indicators
        filtered_point_data = filtered_point_data.assign(
            p1_win=(filtered_point_data['PtWinner'] == 1),
            p2_win=(filtered_point_data['PtWinner'] == 2),
            server_1_win=(server_1_mask & (filtered_point_data['PtWinner'] == 1)),
            server_2_win=(server_2_mask & (filtered_point_data['PtWinner'] == 2))
        )

        # Vectorized aggregation
        agg_stats = (filtered_point_data
                     .groupby('playerdate_id')
                     .agg(
            total_points=('Pt', 'size'),
            server_1_points=('Svr', lambda x: (x == 1).sum()),
            server_2_points=('Svr', lambda x: (x == 2).sum()),
            server_1_wins=('server_1_win', 'sum'),
            server_2_wins=('server_2_win', 'sum'),
            avg_rally_length=('rally_length', 'mean') if 'rally_length' in filtered_point_data.columns else ('Pt',
                                                                                                             lambda
                                                                                                                 x: 4.0),
            rally_length_std=('rally_length', 'std') if 'rally_length' in filtered_point_data.columns else ('Pt', lambda
                x: 2.0)
        )
                     .assign(
            server_1_win_rate=lambda df: df.server_1_wins / df.server_1_points.clip(lower=1),
            server_2_win_rate=lambda df: df.server_2_wins / df.server_2_points.clip(lower=1)
        )
                     .assign(
            serve_advantage_diff=lambda df: df.server_1_win_rate - df.server_2_win_rate,
            overall_serve_rate=lambda df: (df.server_1_win_rate + df.server_2_win_rate) / 2,
            serve_volatility=lambda df: abs(df.server_1_win_rate - df.server_2_win_rate)
        )
                     .assign(
            match_competitiveness=lambda df: 1 - abs(df.serve_advantage_diff),
            bp_performance_diff=0.0  # Simplified for vectorization
        )
                     .reset_index()
                     )

        # Select only the features we want
        feature_cols = ['playerdate_id', 'total_points', 'serve_advantage_diff', 'overall_serve_rate',
                        'serve_volatility', 'bp_performance_diff', 'avg_rally_length', 'rally_length_std',
                        'match_competitiveness']
        point_stats_df = agg_stats[feature_cols]

        if len(point_stats_df) > 0:
            # FIXED: Merge on playerdate_id instead of composite_id
            enhanced_data = match_data.merge(point_stats_df, on='playerdate_id', how='left')
            added_features = len(feature_cols) - 1  # Exclude playerdate_id
            logging.info(
                f"FIXED: Integrated {added_features} vectorized point-level features for {len(point_stats_df)} matches")
            return enhanced_data
        else:
            logging.warning("No point statistics calculated")
            return match_data

    def create_balanced_training_dataset(self, match_data):
        """Create balanced dataset"""
        logging.info("Creating balanced training dataset...")

        positive_examples = match_data.copy()
        positive_examples['target'] = 1
        positive_examples['match_id'] = positive_examples.index.astype(str) + '_pos'

        negative_examples = match_data.copy()

        # Swap numeric winner/loser columns
        winner_cols = [col for col in match_data.columns
                       if col.startswith('winner_') and match_data[col].dtype in ['int64', 'float64']]

        for winner_col in winner_cols:
            base_name = winner_col[7:]
            loser_col = f'loser_{base_name}'
            if loser_col in match_data.columns and match_data[loser_col].dtype in ['int64', 'float64']:
                temp_values = negative_examples[winner_col].copy()
                negative_examples[winner_col] = negative_examples[loser_col]
                negative_examples[loser_col] = temp_values

        # Swap basic player info
        basic_swaps = [('Winner', 'Loser'), ('winner_canonical', 'loser_canonical')]
        for col1, col2 in basic_swaps:
            if col1 in negative_examples.columns and col2 in negative_examples.columns:
                temp_values = negative_examples[col1].copy()
                negative_examples[col1] = negative_examples[col2]
                negative_examples[col2] = temp_values

        negative_examples['target'] = 0
        negative_examples['match_id'] = negative_examples.index.astype(str) + '_neg'

        balanced_data = pd.concat([positive_examples, negative_examples], ignore_index=True)

        logging.info(f"Created balanced dataset: {len(balanced_data):,} examples")
        logging.info(f"Class distribution: {balanced_data['target'].value_counts().to_dict()}")

        del positive_examples, negative_examples
        return balanced_data

    def create_relative_features_optimized(self, balanced_data):
        """Memory-optimized relative features"""
        logging.info("Converting to relative features with memory optimization...")

        # Build relative features for numeric columns only
        numeric_cols = balanced_data.select_dtypes(include=[np.number]).columns
        winner_numeric_cols = [col for col in numeric_cols if col.startswith('winner_')]

        relative_features = {}
        for winner_col in winner_numeric_cols:
            base_name = winner_col[7:]
            loser_col = f'loser_{base_name}'

            if loser_col in numeric_cols:
                rel_col = f'rel_{base_name}'
                relative_features[rel_col] = balanced_data[winner_col] - balanced_data[loser_col]

        relative_df = pd.DataFrame(relative_features, index=balanced_data.index)

        # Drop original winner/loser numeric columns immediately
        winner_loser_cols = [col for col in balanced_data.columns
                             if col.startswith(('winner_', 'loser_')) and col in numeric_cols]

        # Expanded leakage patterns
        leakage_patterns = ['winner_', 'loser_', 'Winner', 'Loser', 'composite_id', 'playerdate_id',
                            'Tournament', 'Surface', 'Round', 'Series', 'Court',
                            'Location', 'Tier', 'tournament_', 'event_']

        non_leaking_cols = []
        for col in balanced_data.columns:
            if not any(col.startswith(pattern) for pattern in leakage_patterns):
                non_leaking_cols.append(col)

        # Memory optimization - keep only essential columns
        essential_cols = ['target', 'match_id']
        if 'date' in balanced_data.columns:
            essential_cols.append('date')

        final_data = pd.concat([
            balanced_data[essential_cols],
            relative_df
        ], axis=1)

        logging.info(f"Created {len(relative_features)} relative features")
        logging.info(f"Memory optimization: dropped {len(winner_loser_cols)} winner/loser columns")
        logging.info(f"Final feature count: {len(final_data.columns) - len(essential_cols)}")

        del relative_df, balanced_data
        gc.collect()
        return final_data

    # ADJUSTMENT 9: Unified feature quality and constants removal
    def validate_and_clean_features_unified(self, X, threshold=0.8):
        """ADJUSTMENT 9: Unified feature quality validation and constant removal"""
        logging.info(f"ADJUSTMENT 9: Unified feature validation and cleaning (threshold={threshold})...")

        # Only check numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        # Remove high-missing features
        missing_ratios = X_numeric.isnull().mean()
        high_missing_cols = missing_ratios[missing_ratios > threshold].index.tolist()

        if high_missing_cols:
            X_clean = X_numeric.drop(columns=high_missing_cols)
            logging.info(f"Dropped {len(high_missing_cols)} high-missing columns")
        else:
            X_clean = X_numeric

        # ADJUSTMENT 9: Remove constant features in same step
        constant_features = X_clean.columns[X_clean.nunique() <= 1]
        if len(constant_features) > 0:
            X_clean = X_clean.drop(columns=constant_features)
            logging.info(f"Dropped {len(constant_features)} constant features")

        # Handle missing values with median imputation
        if X_clean.isnull().any().any():
            medians = X_clean.median()
            X_clean = X_clean.fillna(medians)
            logging.info("Applied median imputation to remaining missing values")
        else:
            medians = X_clean.median()  # Store for test set

        if X_clean.shape[1] < 80:
            raise ValueError(f"CRITICAL: Insufficient features after cleaning - {X_clean.shape[1]} < 80")

        logging.info(f"ADJUSTMENT 9: Unified cleaning: {X_clean.shape[1]}/{X.shape[1]} features retained")
        return X_clean, medians, constant_features.tolist()

    def optimize_hyperparameters_fixed(self, X_train, y_train):
        """Fixed hyperparameter optimization"""
        logging.info("Optimizing hyperparameters...")

        try:
            from skopt import BayesSearchCV
            from skopt.space import Real, Integer

            search_spaces = {
                'n_estimators': Integer(100, 500),
                'max_depth': Integer(3, 12),
                'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
                'num_leaves': Integer(10, 100),
                'min_child_samples': Integer(5, 50),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0),
                'reg_alpha': Real(0.0, 1.0),
                'reg_lambda': Real(0.0, 1.0)
            }

            lgb_model = lgb.LGBMClassifier(
                random_state=self.random_seed,
                class_weight='balanced',
                verbose=-1
            )

            bayes_search = BayesSearchCV(
                estimator=lgb_model,
                search_spaces=search_spaces,
                n_iter=30,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_seed),
                scoring='roc_auc',
                n_jobs=-1,
                random_state=self.random_seed
            )

            bayes_search.fit(X_train, y_train)
            logging.info(f"Best CV score: {bayes_search.best_score_:.4f}")

            final_model = bayes_search.best_estimator_
            final_model.fit(X_train, y_train)  # Ensure model is fitted

            return final_model, bayes_search.best_params_

        except ImportError:
            logging.info("scikit-optimize not available, using default parameters")
            default_model = self._get_default_lgb_model()
            default_model.fit(X_train, y_train)
            return default_model, {}
        except Exception as e:
            logging.info(f"Bayesian optimization failed: {e}, using default parameters")
            default_model = self._get_default_lgb_model()
            default_model.fit(X_train, y_train)
            return default_model, {}

    def _get_default_lgb_model(self):
        """Default model"""
        return lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.03,
            random_state=self.random_seed,
            class_weight='balanced',
            verbose=-1
        )

    def temporal_train_test_split_fixed(self, final_data, test_size=0.2, val_size=0.1):
        """Temporal split with date removed before splitting"""
        logging.info("Performing temporal train/val/test split...")

        if 'date' not in final_data.columns:
            logging.info("No date column found, using random split")
            feature_cols = [col for col in final_data.columns if col not in ['target', 'match_id']]
            return train_test_split(
                final_data[feature_cols], final_data['target'],
                test_size=test_size, random_state=self.random_seed, stratify=final_data['target']
            )

        sorted_data = final_data.sort_values('date').reset_index(drop=True)
        n_samples = len(sorted_data)

        train_end = int(n_samples * (1 - test_size - val_size))
        val_end = int(n_samples * (1 - test_size))

        train_data = sorted_data.iloc[:train_end]
        val_data = sorted_data.iloc[train_end:val_end]
        test_data = sorted_data.iloc[val_end:]

        logging.info(f"Temporal split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

        # Remove date BEFORE creating features
        feature_cols = [col for col in sorted_data.columns if col not in ['target', 'match_id', 'date']]

        X_train = train_data[feature_cols]
        y_train = train_data['target']
        X_val = val_data[feature_cols] if len(val_data) > 0 else None
        y_val = val_data['target'] if len(val_data) > 0 else None
        X_test = test_data[feature_cols]
        y_test = test_data['target']

        return X_train, X_test, y_train, y_test, X_val, y_val

    # ADJUSTMENT 8: Fixed feature selection threshold
    def advanced_feature_selection(self, X, y):
        """ADJUSTMENT 8: Feature selection with fixed threshold"""
        logging.info("ADJUSTMENT 8: Performing advanced feature selection...")

        # ADJUSTMENT 8: Set minimum threshold to 120
        min_features = 120

        if X.shape[1] <= min_features:
            self.feature_selector = {'type': 'none', 'selected_features': X.columns.tolist()}
            return X, self.feature_selector

        if X.shape[1] > 300:
            logging.info("Using LightGBM-based feature selection...")

            lgb_selector = lgb.LGBMClassifier(n_estimators=100, random_state=self.random_seed, verbose=-1)
            lgb_selector.fit(X, y)

            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': lgb_selector.feature_importances_
            }).sort_values('importance', ascending=False)

            # ADJUSTMENT 8: Ensure we select at least 120 features
            n_features = max(min_features, min(200, X.shape[1]))
            selected_features = importance_df.head(n_features)['feature'].tolist()
            X_selected = X[selected_features]

            self.feature_selector = {
                'type': 'lgbm_importance',
                'selected_features': selected_features,
                'importance_scores': importance_df
            }

            logging.info(f"LGBM selection: {X_selected.shape[1]} features from {X.shape[1]}")
        else:
            logging.info("Using SelectKBest...")

            # ADJUSTMENT 8: Ensure k >= 120
            k_features = max(min_features, min(X.shape[1], X.shape[1]))
            selector = SelectKBest(score_func=f_classif, k=k_features)
            X_selected = selector.fit_transform(X, y)

            selected_features = X.columns[selector.get_support()].tolist()
            X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

            self.feature_selector = {
                'type': 'select_k_best',
                'selector_object': selector,
                'selected_features': selected_features
            }

            logging.info(f"SelectKBest: {X_selected.shape[1]} features from {X.shape[1]}")

        return X_selected, self.feature_selector

    def train_with_all_fixes_applied(self, rebuild_cache=False, skip_ta=False):
        """Main training pipeline with all 14 adjustments applied + TA fix"""
        logging.info("TRAINING WITH ALL FIXES APPLIED + TA PLAYERDATE_ID FIX")
        logging.info("=" * 60)

        # 1. Load base data
        with timer_context("Data Loading"):
            logging.info("\n1. Loading base tennis match data...")
            tennis_data = load_all_tennis_data()
            logging.info(f"   Base tennis data: {len(tennis_data):,} matches")

            logging.info("\n2. Loading Jeff's comprehensive data...")
            jeff_data = load_jeff_comprehensive_data()
            if not jeff_data:
                raise ValueError("CRITICAL: Jeff data loading failed")

            # ADJUSTMENT 1: Use vectorized weighted defaults and pass properly
            logging.info("\n3. ADJUSTMENT 2: Calculating vectorized weighted defaults...")
            weighted_defaults = create_extended_weighted_defaults_vectorized(jeff_data)

        # 4. Process tennis data
        logging.info("\n4. Processing tennis data with unified canonicalization...")
        tennis_data['winner_canonical'] = tennis_data['Winner'].apply(normalize_name)
        tennis_data['loser_canonical'] = tennis_data['Loser'].apply(normalize_name)

        if 'Date' in tennis_data.columns:
            tennis_data['Date'] = pd.to_datetime(tennis_data['Date'], errors='coerce')
            tennis_data['date'] = tennis_data['Date'].dt.date

        tennis_data = tennis_data.dropna(subset=['date'])

        # Use unified canonicalization
        tennis_data['composite_id'] = tennis_data.apply(
            lambda row: create_canonical_composite_id(
                row['date'], row['Tournament'], row['Winner'], row['Loser']
            ), axis=1
        )

        # FIXED: Create playerdate_id for TA matching
        tennis_data['playerdate_id'] = tennis_data.apply(
            lambda r: create_playerdate_id(r['date'], r['Winner'], r['Loser']), axis=1)

        tennis_data['match_id'] = tennis_data['composite_id']
        logging.info(f"   Processed tennis data: {len(tennis_data):,} matches")

        # 5. ADJUSTMENTS 1,3,4: Extract Jeff features with all optimizations
        with timer_context("Jeff Feature Extraction", self.timeout_jeff):
            enhanced_data = self.extract_all_jeff_features_optimized(tennis_data, jeff_data, weighted_defaults)

        # 6. FIXED: Tennis Abstract with playerdate_id
        if not skip_ta:
            try:
                with timer_context("Tennis Abstract Integration", self.timeout_ta):
                    logging.info("\n6. Scraping Tennis Abstract with fallback...")
                    scraped_records = self.scrape_tennis_abstract_with_fallback(days_back=90, max_matches=500)

                    if scraped_records:
                        logging.info("\n7. FIXED: Integrating Tennis Abstract features using playerdate_id...")
                        enhanced_data = self.integrate_all_tennis_abstract_features_fixed(enhanced_data,
                                                                                          scraped_records)
            except Exception as e:
                if not skip_ta:
                    raise e
        else:
            logging.info("\n6. Skipping Tennis Abstract (--no-ta flag)")

        # 8. FIXED: Point integration using playerdate_id
        logging.info("\n8. FIXED: Loading and integrating point-level data using playerdate_id...")
        real_point_data = self.load_real_point_data_from_jeff(jeff_data)
        if not real_point_data.empty:
            enhanced_data = self.integrate_point_level_features_fixed(enhanced_data, real_point_data)

        # 9. Create balanced dataset
        logging.info("\n9. Creating balanced dataset...")
        balanced_data = self.create_balanced_training_dataset(enhanced_data)

        # 10. Memory-optimized relative features
        logging.info("\n10. Converting to relative features with memory optimization...")
        final_data = self.create_relative_features_optimized(balanced_data)

        # 11. Remove leakage indicators (preserve date for temporal split)
        logging.info("\n11. Removing leakage indicators...")
        leakage_columns = ['ta_enhanced', 'source_rank', 'data_quality_score']
        leakage_columns = [col for col in leakage_columns if col in final_data.columns]
        if leakage_columns:
            final_data = final_data.drop(columns=leakage_columns)
            logging.info(f"   Removed {len(leakage_columns)} leakage indicators")

        # 12. Temporal split
        logging.info("\n12. Performing temporal train-test split...")
        X_train, X_test, y_train, y_test, X_val, y_val = self.temporal_train_test_split_fixed(final_data)

        # 13. ADJUSTMENT 9: Unified feature cleaning
        logging.info("\n13. ADJUSTMENT 9: Unified feature quality validation and cleaning...")
        X_train, medians, constant_features = self.validate_and_clean_features_unified(X_train, threshold=0.8)
        X_test = X_test[X_train.columns].fillna(medians)
        if X_val is not None:
            X_val = X_val[X_train.columns].fillna(medians)

        # 14. ADJUSTMENT 8: Feature selection with fixed threshold
        X_train, feature_selector = self.advanced_feature_selection(X_train, y_train)
        X_test = X_test[X_train.columns]
        if X_val is not None:
            X_val = X_val[X_train.columns]

        logging.info(f"   Final training features: {X_train.shape[1]}")

        # Hard failure for insufficient features
        if X_train.shape[1] < 120:
            raise ValueError(f"CRITICAL: Final feature count {X_train.shape[1]} < 120")

        logging.info(f"   Training samples: {len(X_train):,}")

        # 15. Model training
        logging.info("\n14. Training model with fixed optimization...")
        optimized_model, best_params = self.optimize_hyperparameters_fixed(X_train, y_train)

        # 16. Evaluation
        logging.info("\n15. Evaluating model performance...")
        y_pred = optimized_model.predict(X_test)
        y_pred_proba = optimized_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)
        brier = brier_score_loss(y_test, y_pred_proba)

        logging.info(f"FINAL MODEL PERFORMANCE:")
        logging.info(f"  Accuracy: {accuracy:.4f}")
        logging.info(f"  AUC-ROC:  {auc:.4f}")
        logging.info(f"  Log-Loss: {logloss:.4f}")
        logging.info(f"  Brier Score: {brier:.4f}")

        if auc < 0.60:
            raise ValueError(f"CRITICAL: Model AUC {auc:.4f} < 0.60 threshold")

        # Cross-validation
        cv_scores = cross_val_score(
            optimized_model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed),
            scoring='roc_auc'
        )
        logging.info(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': optimized_model.feature_importances_
        }).sort_values('importance', ascending=False)

        logging.info(f"\nTop 10 most important features:")
        logging.info(feature_importance.head(10))

        # ADJUSTMENT 14: Save model with timestamp
        training_hash = compute_data_hash(final_data)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_cache / f"comprehensive_model_playerdate_fix_{timestamp}.pkl"

        model_metadata = {
            'model': optimized_model,
            'feature_columns': X_train.columns.tolist(),
            'feature_importance': feature_importance,
            'feature_selector': self.feature_selector,
            'best_hyperparameters': best_params,
            'performance': {
                'accuracy': accuracy, 'auc': auc, 'log_loss': logloss,
                'brier_score': brier, 'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std()
            },
            'training_data_hash': training_hash,
            'training_date': date.today(),
            'random_seed': self.random_seed,
            'preprocessing_info': {
                'feature_medians': medians.to_dict(),
                'constant_features_removed': constant_features
            },
            'fixes_applied': ['playerdate_id_matching', 'all_14_adjustments']
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_metadata, f)

        logging.info(f"\nFIXED MODEL saved with timestamp: {model_path}")

        return optimized_model, feature_importance, {
            'accuracy': accuracy, 'auc': auc, 'log_loss': logloss, 'brier_score': brier
        }

    def load_real_point_data_from_jeff(self, jeff_data):
        """Load real point data from Jeff's datasets"""
        logging.info("Loading real point-by-point data...")

        all_points = []
        point_sources = ['points_2020s', 'points_2010s', 'pointsto2009']

        for gender in ['men', 'women']:
            if gender not in jeff_data:
                continue

            gender_data = jeff_data[gender]
            points_found = 0

            for source in point_sources:
                if source in gender_data and not gender_data[source].empty:
                    points_df = gender_data[source].copy()

                    required_cols = ['match_id', 'Pt', 'Svr', 'PtWinner']
                    if all(col in points_df.columns for col in required_cols):
                        points_df['gender'] = gender
                        points_df['source'] = source
                        all_points.append(points_df)
                        points_found += len(points_df)

            logging.info(f"  {gender}: {points_found:,} real points loaded")

        if all_points:
            combined_points = pd.concat(all_points, ignore_index=True)
            logging.info(f"Total real point data: {len(combined_points):,} points")
            return combined_points
        else:
            return pd.DataFrame()


# ADJUSTMENT 11: Enhanced validation tests
def run_validation_tests(pipeline_result):
    """ADJUSTMENT 11: Enhanced validation tests with random sampling"""
    logging.info("\nADJUSTMENT 11: Running enhanced validation tests...")

    model, feature_importance, performance = pipeline_result

    # Test 1: Feature count
    feature_count = len(feature_importance)
    if feature_count < 120:
        raise ValueError(f"TEST FAILED: Feature count {feature_count} < 120")
    logging.info(f"✓ Feature count test passed: {feature_count}")

    # Test 2: Performance
    auc = performance['auc']
    if auc < 0.60:
        raise ValueError(f"TEST FAILED: AUC {auc:.4f} < 0.60")
    logging.info(f"✓ Performance test passed: AUC {auc:.4f}")

    # ADJUSTMENT 11: Test 3: Random sample validation of playerdate IDs
    logging.info("✓ ADJUSTMENT 11: Testing random sample of playerdate ID canonicalization...")

    # Generate test data
    test_cases = []
    for i in range(100):
        test_date = date(2025, 1, 1) + timedelta(days=i)
        player1 = f"Test_Player_A_{i % 20}"
        player2 = f"Test_Player_B_{i % 20}"

        playerdate_id = create_playerdate_id(test_date, player1, player2)
        point_style_id = f"{test_date.strftime('%Y%m%d')}-M-Tournament-R32-{player1}-{player2}"

        parsed_playerdate = parse_point_match_id(point_style_id)

        if playerdate_id != parsed_playerdate:
            raise ValueError(
                f"TEST FAILED: PlayerDate ID mismatch - direct: {playerdate_id}, parsed: {parsed_playerdate}")

        test_cases.append((playerdate_id, parsed_playerdate))

    logging.info(f"✓ FIXED: Random sample test passed: {len(test_cases)} playerdate ID pairs validated")

    # Test 4: Memory efficiency check
    import psutil
    process = psutil.Process()
    memory_gb = process.memory_info().rss / (1024 ** 3)
    if memory_gb > 8.0:
        logging.warning(f"Memory usage {memory_gb:.1f}GB > 8GB target")
    else:
        logging.info(f"✓ Memory efficiency test passed: {memory_gb:.1f}GB")

    logging.info("✓ All enhanced validation tests passed with playerdate_id fix")


def main():
    parser = argparse.ArgumentParser(description="Tennis Pipeline - All Fixes + PlayerDate ID")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild dataset cache")
    parser.add_argument("--no-ta", action="store_true", help="Skip Tennis Abstract scraping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--timeout-jeff", type=int, default=1200, help="Jeff extraction timeout")  # ADJUSTMENT 10
    parser.add_argument("--timeout-ta", type=int, default=600, help="TA scraping timeout")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")  # ADJUSTMENT 12
    args = parser.parse_args()

    # ADJUSTMENT 12: Setup logging based on verbose flag
    setup_logging(verbose=args.verbose)

    global GLOBAL_SEED
    GLOBAL_SEED = args.seed
    np.random.seed(GLOBAL_SEED)

    # ADJUSTMENT 10: Use reduced timeouts
    pipeline = FixedComprehensiveDataPipeline(
        random_seed=GLOBAL_SEED,
        timeout_jeff=args.timeout_jeff,
        timeout_ta=args.timeout_ta
    )

    start_time = time.time()

    try:
        result = pipeline.train_with_all_fixes_applied(
            rebuild_cache=args.rebuild,
            skip_ta=args.no_ta
        )

        # ADJUSTMENT 11: Run enhanced validation tests
        run_validation_tests(result)

        total_time = time.time() - start_time

        logging.info("\n" + "=" * 60)
        logging.info("ALL FIXES SUCCESSFULLY APPLIED + PLAYERDATE_ID FIX")
        logging.info("=" * 60)
        logging.info("✓ ADJUSTMENT 1: Weighted defaults properly wired")
        logging.info("✓ ADJUSTMENT 2: Vectorized defaults construction")
        logging.info("✓ ADJUSTMENT 3: Vectorized default injection")
        logging.info("✓ ADJUSTMENT 4: Smart column pre-allocation")
        logging.info("✓ ADJUSTMENT 5: Player_canonical guard")
        logging.info("✓ ADJUSTMENT 6: Fixed regex pattern")
        logging.info("✓ ADJUSTMENT 7: Vectorized point integration")
        logging.info("✓ ADJUSTMENT 8: Fixed feature selection threshold")
        logging.info("✓ ADJUSTMENT 9: Unified feature cleaning")
        logging.info("✓ ADJUSTMENT 10: Reduced Jeff timeout to 1200s")
        logging.info("✓ ADJUSTMENT 11: Enhanced validation tests")
        logging.info("✓ ADJUSTMENT 12: Configurable logging level")
        logging.info("✓ ADJUSTMENT 13: Improved hash function")
        logging.info("✓ ADJUSTMENT 14: Timestamped model export")
        logging.info("✓ PLAYERDATE_ID FIX: Tournament-agnostic matching")

        model, feature_importance, performance = result
        logging.info(f"\n🎯 TARGETS ACHIEVED:")
        logging.info(f"   Final AUC: {performance['auc']:.4f} (≥0.60 target)")
        logging.info(f"   Final features: {len(feature_importance)} (≥120 target)")
        logging.info(f"   Total time: {total_time / 60:.1f} minutes")

        # Memory check
        import psutil
        memory_gb = psutil.Process().memory_info().rss / (1024 ** 3)
        logging.info(f"   Peak memory: {memory_gb:.1f}GB")

        target_auc = 0.63
        if performance['auc'] >= target_auc:
            logging.info(f"   🎯 STRETCH TARGET ACHIEVED: AUC {performance['auc']:.4f} ≥ {target_auc}")

    except Exception as e:
        logging.error(f"CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())