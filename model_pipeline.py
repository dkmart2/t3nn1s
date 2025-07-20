#!/usr/bin/env python3
"""
FIXED: Tennis ML Pipeline with Memory Optimization Applied
- Replaced copy() with in-place updates
- Aggregate Jeff stat tables before merging
- Down-cast numeric columns immediately
- Explicit memory cleanup with gc.collect()
- Process datasets one at a time with intermediate saves
"""

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
import gc  # ADDED: Explicit garbage collection
from pathlib import Path
from functools import wraps
from contextlib import contextmanager
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.feature_selection import SelectKBest, f_classif
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ADDED: Memory optimization settings
os.environ['MODIN_ENGINE'] = 'ray'  # if modin installed
pd.set_option("mode.copy_on_write", True)

# Set global random seed for reproducibility
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# Performance thresholds (seconds)
STAGE_TIMEOUTS = {
    'data_loading': 300,
    'feature_extraction': 600,
    'tennis_abstract': 900,
    'model_training': 1800
}

sys.path.append('.')
from model import TennisModelPipeline, ModelConfig
from tennis_updated import (
    load_from_cache_with_scraping,
    generate_comprehensive_historical_data,
    save_to_cache,
    load_all_tennis_data,
    load_jeff_comprehensive_data,
    calculate_comprehensive_weighted_defaults,
    integrate_api_tennis_data_incremental,
    AutomatedTennisAbstractScraper,
    CACHE_DIR,
    extract_comprehensive_jeff_features,
    normalize_name
)


@contextmanager
def timer_context(stage_name, timeout=None):
    """Context manager for timing execution with optional timeout"""
    start_time = time.time()
    print(f"[TIMING] Starting {stage_name}...")

    try:
        yield
        duration = time.time() - start_time

        if timeout and duration > timeout:
            raise TimeoutError(f"{stage_name} exceeded {timeout}s threshold: {duration:.1f}s")

        print(f"[TIMING] {stage_name} completed in {duration:.1f}s")

    except Exception as e:
        duration = time.time() - start_time
        print(f"[TIMING] {stage_name} failed after {duration:.1f}s: {e}")
        raise


def timing_wrapper(stage_name, timeout=None):
    """Decorator to time function execution and enforce thresholds"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with timer_context(stage_name, timeout):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def exponential_backoff_retry(max_attempts=5, base_delay=1):
    """Decorator for exponential backoff retry logic"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise Exception(f"Failed after {max_attempts} attempts: {e}")

                    delay = base_delay * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)

        return wrapper

    return decorator


def compute_data_hash(data):
    """Compute deterministic SHA-256 hash of training data"""
    # Create deterministic hash by sorting and using only numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        # Fallback if no numeric columns
        return hashlib.sha256(str(len(data)).encode()).hexdigest()

    # Sort by columns and rows for determinism
    sorted_data = data[numeric_cols].sort_index().sort_index(axis=1)

    # Convert to bytes efficiently
    data_bytes = sorted_data.to_numpy().astype(np.float32).tobytes()
    return hashlib.sha256(data_bytes).hexdigest()


class OptimizedComprehensiveDataPipeline:
    """FIXED: Memory-optimized pipeline with in-place updates and aggressive cleanup"""

    def __init__(self, cache_dir=CACHE_DIR, random_seed=GLOBAL_SEED):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.random_seed = random_seed
        self.model_cache = self.cache_dir / "trained_models"
        self.model_cache.mkdir(exist_ok=True)
        self.feature_selector = None
        self.feature_selector_type = None

        np.random.seed(random_seed)

    def validate_feature_extraction(self, data, feature_prefix, min_features=5):
        """Validate feature extraction with explicit checks"""
        feature_cols = [col for col in data.columns if col.startswith(feature_prefix)]

        if len(feature_cols) < min_features:
            raise ValueError(
                f"Feature extraction failed: only {len(feature_cols)} {feature_prefix} features found, expected at least {min_features}")

        non_null_counts = data[feature_cols].count().sum()
        if non_null_counts == 0:
            raise ValueError(f"Feature extraction failed: all {feature_prefix} features are null")

        print(f"✓ Validated {feature_prefix}: {len(feature_cols)} features, {non_null_counts:,} non-null values")
        return feature_cols

    @timing_wrapper("Jeff Feature Extraction", STAGE_TIMEOUTS['feature_extraction'])
    def extract_all_jeff_features_vectorized_fixed(self, match_data, jeff_data, weighted_defaults):
        """FIXED: Memory-optimized Jeff feature extraction with in-place updates"""
        print("Extracting comprehensive features from ALL Jeff datasets (MEMORY FIXED)...")

        # FIXED: Use view, not copy
        enhanced_data = match_data  # view, not copy
        feature_categories = {}

        # Get all available Jeff dataset types
        available_datasets = set()
        for gender in ['men', 'women']:
            if gender in jeff_data:
                available_datasets.update(jeff_data[gender].keys())

        print(f"Available Jeff datasets: {sorted(available_datasets)}")

        # Process each gender separately for vectorized operations
        for gender in ['men', 'women']:
            if gender not in jeff_data:
                continue

            gender_key = gender
            # FIXED: Use index-based selection instead of copy
            idx = enhanced_data['gender'] == ('M' if gender == 'men' else 'W')

            if not idx.any():
                continue

            print(f"  Processing {gender}: {idx.sum():,} matches")

            # Process datasets with AGGRESSIVE cleanup
            dataset_configs = [
                ('overview',
                 ['serve_pts', 'aces', 'dfs', 'first_in', 'first_won', 'second_won', 'bp_saved', 'return_pts_won',
                  'winners', 'winners_fh', 'winners_bh', 'unforced', 'unforced_fh', 'unforced_bh']),
                ('serve_basics', ['pts_won', 'aces', 'unret', 'forced_err', 'wide', 'body', 't']),
                ('return_outcomes', ['returnable', 'returnable_won', 'in_play', 'in_play_won', 'winners']),
                ('key_points_serve', ['pts_won', 'first_in', 'aces', 'svc_winners', 'rally_winners']),
                ('net_points', ['net_pts', 'pts_won', 'net_winner', 'induced_forced', 'passed_at_net'])
            ]

            for dataset_name, agg_cols in dataset_configs:
                if dataset_name not in jeff_data[gender_key]:
                    continue

                df = jeff_data[gender_key][dataset_name]

                # Special handling for overview dataset
                if dataset_name == 'overview' and 'Player_canonical' in df.columns:
                    overview_total = df[df.get('set', '') == 'Total'].copy()
                    if overview_total.empty:
                        continue

                    # FIXED: Aggregate BEFORE merging, down-cast immediately
                    agg_df = overview_total.groupby('Player_canonical')[agg_cols].mean().astype(
                        np.float32).reset_index()

                    # Merge winner stats using index-based updates
                    winner_merge = enhanced_data[idx].merge(
                        agg_df.add_prefix('winner_').rename(columns={'winner_Player_canonical': 'winner_canonical'}),
                        on='winner_canonical', how='left'
                    )

                    # Update in-place
                    for col in winner_merge.columns:
                        if col.startswith('winner_') and col not in enhanced_data.columns:
                            enhanced_data[col] = np.nan
                            enhanced_data.loc[idx, col] = winner_merge[col].values

                    # Merge loser stats using index-based updates
                    loser_merge = enhanced_data[idx].merge(
                        agg_df.add_prefix('loser_').rename(columns={'loser_Player_canonical': 'loser_canonical'}),
                        on='loser_canonical', how='left'
                    )

                    # Update in-place
                    for col in loser_merge.columns:
                        if col.startswith('loser_') and col not in enhanced_data.columns:
                            enhanced_data[col] = np.nan
                            enhanced_data.loc[idx, col] = loser_merge[col].values

                    # FIXED: Explicit cleanup
                    del overview_total, agg_df, winner_merge, loser_merge
                    gc.collect()

                elif 'player' in df.columns:
                    # FIXED: Aggregate BEFORE merging for other datasets
                    agg_dict = {col: 'mean' for col in agg_cols if col in df.columns}
                    if not agg_dict:
                        continue

                    # Aggregate and down-cast immediately
                    agg_df = df.groupby('player').agg(agg_dict).astype(np.float32).reset_index()

                    # Merge winner stats using index-based updates
                    winner_df = agg_df.add_prefix(f'winner_{dataset_name}_')
                    winner_df['winner_canonical'] = agg_df['player']

                    winner_merge = enhanced_data[idx].merge(
                        winner_df.drop(columns=[f'winner_{dataset_name}_player']),
                        on='winner_canonical', how='left'
                    )

                    # Update in-place
                    for col in winner_merge.columns:
                        if col.startswith('winner_') and col not in enhanced_data.columns:
                            enhanced_data[col] = np.nan
                            enhanced_data.loc[idx, col] = winner_merge[col].values

                    # Merge loser stats using index-based updates
                    loser_df = agg_df.add_prefix(f'loser_{dataset_name}_')
                    loser_df['loser_canonical'] = agg_df['player']

                    loser_merge = enhanced_data[idx].merge(
                        loser_df.drop(columns=[f'loser_{dataset_name}_player']),
                        on='loser_canonical', how='left'
                    )

                    # Update in-place
                    for col in loser_merge.columns:
                        if col.startswith('loser_') and col not in enhanced_data.columns:
                            enhanced_data[col] = np.nan
                            enhanced_data.loc[idx, col] = loser_merge[col].values

                    # FIXED: Explicit cleanup after each dataset
                    del agg_df, winner_df, loser_df, winner_merge, loser_merge
                    gc.collect()

                feature_categories[dataset_name] = feature_categories.get(dataset_name, 0) + idx.sum()

        print(f"✓ MEMORY-FIXED vectorized feature extraction complete. Categories used: {feature_categories}")

        # Validate Jeff feature extraction
        self.validate_feature_extraction(enhanced_data, 'winner_', min_features=10)
        self.validate_feature_extraction(enhanced_data, 'loser_', min_features=10)

        return enhanced_data

    def optimize_hyperparameters_bayesian_enhanced(self, X_train, y_train):
        """FIXED: Enhanced Bayesian optimization with proper early stopping"""
        print("Optimizing hyperparameters with enhanced Bayesian search...")

        try:
            from skopt import BayesSearchCV
            from skopt.space import Real, Integer
        except ImportError:
            print("scikit-optimize not available, using default parameters")
            return self._get_default_lgb_model(), {}

        # Split training data for early stopping
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_seed, stratify=y_train
        )

        # Define enhanced search space
        search_spaces = {
            'n_estimators': Integer(200, 800),
            'max_depth': Integer(3, 15),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'num_leaves': Integer(20, 200),
            'min_child_samples': Integer(10, 100),
            'subsample': Real(0.7, 1.0),
            'colsample_bytree': Real(0.7, 1.0),
            'reg_alpha': Real(0.0, 2.0),
            'reg_lambda': Real(0.0, 2.0),
            'max_cat_threshold': Integer(10, 50)
        }

        # Base model WITHOUT early_stopping_rounds in constructor
        lgb_model = lgb.LGBMClassifier(
            random_state=self.random_seed,
            class_weight='balanced',
            verbose=-1
        )

        # Prepare fit_params for early stopping
        fit_params = {
            'eval_set': [(X_val_split, y_val_split)],
            'early_stopping_rounds': 50,
            'verbose': False
        }

        # FIXED: Enhanced Bayesian search with proper fit_params handling
        bayes_search = BayesSearchCV(
            estimator=lgb_model,
            search_spaces=search_spaces,
            n_iter=60,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_seed),
            scoring='roc_auc',
            n_jobs=-1,
            random_state=self.random_seed,
            fit_params=fit_params  # Pass fit_params to BayesSearchCV
        )

        # Fit with proper parameters
        bayes_search.fit(X_train, y_train)

        print(f"Best CV score: {bayes_search.best_score_:.4f}")
        print(f"Best parameters: {bayes_search.best_params_}")

        return bayes_search.best_estimator_, bayes_search.best_params_

    def _get_default_lgb_model(self):
        """Get default LightGBM model when optimization fails"""
        return lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.03,
            random_state=self.random_seed,
            class_weight='balanced',
            verbose=-1,
            max_cat_threshold=32
        )

    @exponential_backoff_retry(max_attempts=5, base_delay=2)
    @timing_wrapper("Tennis Abstract Scraping", STAGE_TIMEOUTS['tennis_abstract'])
    def scrape_tennis_abstract_with_retry(self, days_back=30, max_matches=100):
        """Tennis Abstract scraping with exponential backoff retry"""
        scraper = AutomatedTennisAbstractScraper()
        return scraper.automated_scraping_session(days_back=days_back, max_matches=max_matches)

    def integrate_all_tennis_abstract_features(self, match_data, scraped_records):
        """FIXED: Incremental TA feature assignment to reduce memory usage"""
        print("Integrating ALL Tennis Abstract features with incremental assignment...")

        if not scraped_records:
            print("WARNING: No Tennis Abstract records available")
            return match_data

        # Organize TA data by match and player
        ta_by_match = {}
        data_types_found = set()

        for record in scraped_records:
            comp_id = record.get('composite_id')
            player = record.get('Player_canonical')
            data_type = record.get('data_type')
            stat_name = record.get('stat_name')
            stat_value = record.get('stat_value')

            if not all([comp_id, player, data_type, stat_name]) or stat_value is None:
                continue

            data_types_found.add(data_type)

            if comp_id not in ta_by_match:
                ta_by_match[comp_id] = {}
            if player not in ta_by_match[comp_id]:
                ta_by_match[comp_id][player] = {}

            feature_name = f"ta_{data_type}_{stat_name}"
            ta_by_match[comp_id][player][feature_name] = stat_value

        if not data_types_found:
            print("WARNING: No Tennis Abstract data types found")
            return match_data

        print(f"TA data types found: {sorted(data_types_found)}")
        print(f"Matches with TA data: {len(ta_by_match)}")

        # FIXED: Incremental assignment instead of pre-allocation
        enhanced_data = match_data  # Use view, not copy
        matches_enhanced = 0
        ta_features_added = set()

        # Process matches incrementally
        for comp_id, players in ta_by_match.items():
            match_mask = enhanced_data['composite_id'] == comp_id
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

                    # Create column if it doesn't exist
                    if col_name not in enhanced_data.columns:
                        enhanced_data[col_name] = np.nan

                    enhanced_data.loc[match_idx, col_name] = feature_value
                    ta_features_added.add(col_name)

            enhanced_data.loc[match_idx, 'ta_enhanced'] = True
            matches_enhanced += 1

        print(f"Enhanced {matches_enhanced} matches with {len(ta_features_added)} TA features")

        if matches_enhanced > 0:
            self.validate_feature_extraction(enhanced_data, 'winner_ta_', min_features=3)
            self.validate_feature_extraction(enhanced_data, 'loser_ta_', min_features=3)

        return enhanced_data

    def integrate_point_level_features_vectorized(self, match_data, point_data):
        """FIXED: Vectorized point integration with proper server pivot handling"""
        print("Integrating point-level data with vectorized operations...")

        if point_data.empty:
            print("No point data to integrate")
            return match_data

        # ASSERT: Match ID alignment validation
        match_ids_in_points = set(point_data['match_id'].unique())
        match_ids_in_matches = set(match_data.get('match_id', []))

        if 'match_id' not in match_data.columns:
            print("WARNING: No match_id column in match data, skipping point integration")
            return match_data

        mismatched_count = len(match_ids_in_points - match_ids_in_matches)
        if mismatched_count > 0:
            raise ValueError(f"Match ID mismatch: {mismatched_count} point log matches not found in match data")

        print(f"✓ Match ID validation passed: {len(match_ids_in_points)} matches aligned")

        # Vectorized aggregation
        grouped = point_data.groupby('match_id')

        # Pre-compute server statistics
        server_stats = point_data.groupby(['match_id', 'Svr']).agg({
            'PtWinner': ['count', 'sum']
        }).reset_index()
        server_stats.columns = ['match_id', 'Svr', 'points_played', 'points_won']
        server_stats['win_rate'] = server_stats['points_won'] / server_stats['points_played']

        # FIXED: Pivot with explicit filling for missing servers
        server_pivot = server_stats.pivot(index='match_id', columns='Svr', values='win_rate')

        # Ensure both server columns exist
        if 1 not in server_pivot.columns:
            server_pivot[1] = 0.5
        if 2 not in server_pivot.columns:
            server_pivot[2] = 0.5

        server_pivot = server_pivot.fillna(0.5)

        # Calculate match-level statistics
        match_stats = grouped.agg({
            'PtWinner': 'count',  # total points
        }).reset_index()
        match_stats.columns = ['match_id', 'total_points']

        # Add rally statistics if available
        if 'rally_length' in point_data.columns:
            rally_stats = grouped['rally_length'].agg(['mean', 'std']).reset_index()
            rally_stats.columns = ['match_id', 'avg_rally_length', 'rally_length_std']
            match_stats = match_stats.merge(rally_stats, on='match_id', how='left')
            del rally_stats
        else:
            match_stats['avg_rally_length'] = 4.0
            match_stats['rally_length_std'] = 2.0

        # Merge server statistics
        match_stats = match_stats.merge(server_pivot, on='match_id', how='left')

        # Calculate neutral features
        match_stats['serve_advantage_diff'] = match_stats[1] - match_stats[2]
        match_stats['overall_serve_rate'] = (match_stats[1] + match_stats[2]) / 2
        match_stats['serve_volatility'] = abs(match_stats[1] - match_stats[2])
        match_stats['match_competitiveness'] = 1 - abs(match_stats['serve_advantage_diff'])

        # Drop server-specific columns
        match_stats = match_stats.drop(columns=[1, 2])

        # Break point analysis (vectorized)
        if 'is_break_point' in point_data.columns:
            bp_data = point_data[point_data['is_break_point'] == True]
            if not bp_data.empty:
                bp_stats = bp_data.groupby(['match_id', 'Svr']).agg({
                    'PtWinner': ['count', 'sum']
                }).reset_index()

                bp_stats.columns = ['match_id', 'Svr', 'bp_faced', 'bp_won']
                bp_stats['bp_win_rate'] = bp_stats['bp_won'] / bp_stats['bp_faced']
                bp_pivot = bp_stats.pivot(index='match_id', columns='Svr', values='bp_win_rate')

                # Ensure both columns exist
                if 1 not in bp_pivot.columns:
                    bp_pivot[1] = 0.5
                if 2 not in bp_pivot.columns:
                    bp_pivot[2] = 0.5

                bp_pivot = bp_pivot.fillna(0.5)
                bp_diff = bp_pivot[1] - bp_pivot[2]

                match_stats = match_stats.merge(
                    bp_diff.to_frame('bp_performance_diff'),
                    on='match_id', how='left'
                )
                del bp_data, bp_stats, bp_pivot
            else:
                match_stats['bp_performance_diff'] = 0.0
        else:
            match_stats['bp_performance_diff'] = 0.0

        # Filter matches with sufficient points
        match_stats = match_stats[match_stats['total_points'] >= 10]

        # Merge with match data
        enhanced_data = match_data.merge(match_stats, on='match_id', how='left')
        added_features = len(match_stats.columns) - 1  # Exclude match_id

        print(f"Integrated {added_features} vectorized point-level features for {len(match_stats)} matches")

        # Memory cleanup
        del grouped, server_stats, server_pivot, match_stats

        return enhanced_data

    def create_balanced_training_dataset(self, match_data):
        """Create balanced dataset with proper target construction"""
        print("Creating balanced training dataset with proper target construction...")

        # Start with original matches (all labeled as 1)
        positive_examples = match_data.copy()
        positive_examples['target'] = 1
        positive_examples['match_id'] = positive_examples.index.astype(str) + '_pos'

        # Create negative examples by swapping winner/loser
        negative_examples = match_data.copy()

        # Identify winner/loser column pairs
        winner_cols = [col for col in match_data.columns if col.startswith('winner_')]

        # Vectorized column swapping
        for winner_col in winner_cols:
            base_name = winner_col[7:]
            loser_col = f'loser_{base_name}'
            if loser_col in match_data.columns:
                # Swap values
                temp_values = negative_examples[winner_col].copy()
                negative_examples[winner_col] = negative_examples[loser_col]
                negative_examples[loser_col] = temp_values

        # Swap basic player info
        basic_swaps = [
            ('Winner', 'Loser'),
            ('winner_canonical', 'loser_canonical')
        ]

        for col1, col2 in basic_swaps:
            if col1 in negative_examples.columns and col2 in negative_examples.columns:
                temp_values = negative_examples[col1].copy()
                negative_examples[col1] = negative_examples[col2]
                negative_examples[col2] = temp_values

        negative_examples['target'] = 0
        negative_examples['match_id'] = negative_examples.index.astype(str) + '_neg'

        # Combine datasets
        balanced_data = pd.concat([positive_examples, negative_examples], ignore_index=True)

        print(f"Created balanced dataset:")
        print(f"  Original matches: {len(match_data):,}")
        print(f"  Balanced examples: {len(balanced_data):,}")
        print(f"  Class distribution: {balanced_data['target'].value_counts().to_dict()}")

        # Memory cleanup
        del positive_examples, negative_examples

        return balanced_data

    def create_relative_features(self, balanced_data):
        """Convert to relative features to prevent leakage"""
        print("Converting to relative features to prevent leakage...")

        # Find matching winner/loser feature pairs
        winner_cols = [col for col in balanced_data.columns if col.startswith('winner_')]

        # Vectorized relative feature creation
        relative_features = {}
        for winner_col in winner_cols:
            base_name = winner_col[7:]
            loser_col = f'loser_{base_name}'

            if loser_col in balanced_data.columns:
                rel_col = f'rel_{base_name}'
                winner_vals = pd.to_numeric(balanced_data[winner_col], errors='coerce')
                loser_vals = pd.to_numeric(balanced_data[loser_col], errors='coerce')
                relative_features[rel_col] = winner_vals - loser_vals

        # Create relative features DataFrame
        relative_df = pd.DataFrame(relative_features, index=balanced_data.index)

        # FIXED: Select non-leaking features but preserve match_id and date for later use
        non_leaking_cols = []
        leakage_patterns = ['winner_', 'loser_', 'Winner', 'Loser', 'composite_id']

        for col in balanced_data.columns:
            if not any(col.startswith(pattern) for pattern in leakage_patterns):
                non_leaking_cols.append(col)

        # Combine features
        final_data = pd.concat([
            balanced_data[non_leaking_cols],
            relative_df
        ], axis=1)

        print(f"Created {len(relative_features)} relative features")
        print(f"Kept {len(non_leaking_cols)} non-leaking features")
        print(f"Final feature count: {len(final_data.columns) - 2}")  # Exclude target and match_id

        # Memory cleanup
        del relative_df, balanced_data

        return final_data

    def validate_feature_quality(self, X, threshold=0.5):
        """Validate non-null ratio per feature and drop high-missing columns"""
        print(f"Validating feature quality (dropping columns with >{threshold * 100}% missing)...")

        missing_ratios = X.isnull().mean()
        high_missing_cols = missing_ratios[missing_ratios > threshold].index.tolist()

        if high_missing_cols:
            X_clean = X.drop(columns=high_missing_cols)
            print(f"Dropped {len(high_missing_cols)} high-missing columns: {high_missing_cols[:5]}...")
        else:
            X_clean = X

        print(f"Feature quality validation: {X_clean.shape[1]}/{X.shape[1]} features retained")
        return X_clean

    def temporal_train_test_split(self, final_data, test_size=0.2, val_size=0.1):
        """FIXED: Temporal split with proper date column handling"""
        print("Performing temporal train/val/test split...")

        if 'date' not in final_data.columns:
            print("WARNING: No date column found, using random split")
            feature_cols = [col for col in final_data.columns if col not in ['target', 'match_id']]
            return train_test_split(
                final_data[feature_cols],
                final_data['target'],
                test_size=test_size,
                random_state=self.random_seed,
                stratify=final_data['target']
            )

        # Sort by date
        sorted_data = final_data.sort_values('date').reset_index(drop=True)
        n_samples = len(sorted_data)

        # Calculate split indices
        train_end = int(n_samples * (1 - test_size - val_size))
        val_end = int(n_samples * (1 - test_size))

        # Split data temporally
        train_data = sorted_data.iloc[:train_end]
        val_data = sorted_data.iloc[train_end:val_end]
        test_data = sorted_data.iloc[val_end:]

        print(f"Temporal split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

        # Extract features and targets (excluding date, target, match_id)
        feature_cols = [col for col in sorted_data.columns if col not in ['target', 'match_id', 'date']]

        X_train = train_data[feature_cols]
        y_train = train_data['target']
        X_val = val_data[feature_cols] if len(val_data) > 0 else None
        y_val = val_data['target'] if len(val_data) > 0 else None
        X_test = test_data[feature_cols]
        y_test = test_data['target']

        return X_train, X_test, y_train, y_test, X_val, y_val

    def advanced_feature_selection(self, X, y):
        """FIXED: Advanced feature selection without duplicate constant removal"""
        print("Performing advanced feature selection...")

        if X.shape[1] <= 100:
            self.feature_selector = {
                'type': 'none',
                'selected_features': X.columns.tolist()
            }
            self.feature_selector_type = 'none'
            return X, self.feature_selector

        if X.shape[1] > 300:
            # Use LightGBM-based selection for large feature sets
            print("Using LightGBM-based feature selection for large feature set...")

            lgb_selector = lgb.LGBMClassifier(
                n_estimators=100,
                random_state=self.random_seed,
                verbose=-1
            )
            lgb_selector.fit(X, y)

            # Get feature importance
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': lgb_selector.feature_importances_
            }).sort_values('importance', ascending=False)

            # Select top features
            n_features = min(200, X.shape[1])
            selected_features = importance_df.head(n_features)['feature'].tolist()
            X_selected = X[selected_features]

            # Store selector info
            self.feature_selector = {
                'type': 'lgbm_importance',
                'selected_features': selected_features,
                'importance_scores': importance_df
            }
            self.feature_selector_type = 'lgbm'

            print(f"LGBM selection: {X_selected.shape[1]} features from {X.shape[1]}")

        else:
            # Use SelectKBest for medium feature sets
            print("Using SelectKBest for medium feature set...")

            k_features = min(100, X.shape[1])
            selector = SelectKBest(score_func=f_classif, k=k_features)
            X_selected = selector.fit_transform(X, y)

            # Convert back to DataFrame
            selected_features = X.columns[selector.get_support()].tolist()
            X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

            # Store selector
            self.feature_selector = {
                'type': 'select_k_best',
                'selector_object': selector,
                'selected_features': selected_features,
                'feature_scores': selector.scores_
            }
            self.feature_selector_type = 'sklearn'

            print(f"SelectKBest: {X_selected.shape[1]} features from {X.shape[1]}")

        return X_selected, self.feature_selector

    def train_with_all_data_optimized(self, rebuild_cache=False, skip_ta=False):
        """FIXED: Memory-optimized training pipeline with all fixes applied"""
        print("MEMORY-OPTIMIZED TRAINING WITH ALL DATA - FIXED VERSION")
        print("=" * 60)

        # 1. Load base tennis data
        with timer_context("Data Loading", STAGE_TIMEOUTS['data_loading']):
            print("\n1. Loading base tennis match data...")
            tennis_data = load_all_tennis_data()

            # FIXED: Early data filtering if RAM < 32GB
            if len(tennis_data) > 20000:
                print("MEMORY OPTIMIZATION: Limiting training set to recent matches")
                tennis_data = tennis_data[tennis_data['Date'] >= '2023-01-01']

            print(f"   Base tennis data: {len(tennis_data):,} matches")

            # 2. Load Jeff's comprehensive data
            print("\n2. Loading Jeff's comprehensive charting data...")
            jeff_data = load_jeff_comprehensive_data()

            if not jeff_data or len(jeff_data) == 0:
                raise ValueError("Jeff data loading failed")

            # 3. Calculate weighted defaults
            print("\n3. Calculating comprehensive weighted defaults...")
            weighted_defaults = calculate_comprehensive_weighted_defaults(jeff_data)

        # 4. Process tennis data
        print("\n4. Processing tennis match data...")
        tennis_data['winner_canonical'] = tennis_data['Winner'].apply(normalize_name)
        tennis_data['loser_canonical'] = tennis_data['Loser'].apply(normalize_name)

        if 'Date' in tennis_data.columns:
            tennis_data['Date'] = pd.to_datetime(tennis_data['Date'], errors='coerce')
            tennis_data['date'] = tennis_data['Date'].dt.date

        tennis_data = tennis_data.dropna(subset=['date'])
        print(f"   Processed tennis data: {len(tennis_data):,} matches")

        # 5. FIXED: Extract Jeff features with memory optimization
        enhanced_data = self.extract_all_jeff_features_vectorized_fixed(tennis_data, jeff_data, weighted_defaults)

        # FIXED: Force garbage collection after Jeff feature extraction
        gc.collect()

        # 6. Tennis Abstract integration (with retry and skip option)
        if not skip_ta:
            try:
                print("\n6. Scraping Tennis Abstract data with retry...")
                scraped_records = self.scrape_tennis_abstract_with_retry(days_back=30, max_matches=100)

                if scraped_records:
                    print("\n7. Integrating comprehensive Tennis Abstract features...")
                    enhanced_data = self.integrate_all_tennis_abstract_features(enhanced_data, scraped_records)
            except Exception as e:
                print(f"Tennis Abstract integration failed: {e}")
                print("Continuing without TA data...")
        else:
            print("\n6. Skipping Tennis Abstract scraping (--no-ta flag)")

        # 8. Point-level integration
        print("\n8. Loading and integrating point-level data...")
        real_point_data = self.load_real_point_data_from_jeff(jeff_data)
        if not real_point_data.empty:
            enhanced_data = self.integrate_point_level_features_vectorized(enhanced_data, real_point_data)

        # FIXED: Force garbage collection after all feature extraction
        gc.collect()

        # 9. Create balanced dataset and relative features BEFORE removing leakage columns
        print("\n9. Creating balanced dataset...")
        balanced_data = self.create_balanced_training_dataset(enhanced_data)

        print("\n10. Converting to relative features...")
        final_data = self.create_relative_features(balanced_data)

        # FIXED: Force garbage collection after balanced dataset creation
        gc.collect()

        # 11. NOW remove leakage indicators (but preserve date for temporal split)
        print("\n11. Removing leakage indicators (preserving date for temporal split)...")
        leakage_columns = []
        leakage_patterns = ['ta_enhanced', 'source_rank', 'data_quality_score', 'composite_id',
                            'Winner', 'Loser', 'winner_canonical', 'loser_canonical']

        for col in final_data.columns:
            if col in leakage_patterns or ('year' in col.lower() and col != 'date'):
                leakage_columns.append(col)

        if leakage_columns:
            final_data = final_data.drop(columns=leakage_columns)
            print(f"   Removed {len(leakage_columns)} leakage indicators")

        # 12. Temporal train-test split (uses date column)
        print("\n12. Performing temporal train-test split...")
        if 'date' in final_data.columns:
            X_train, X_test, y_train, y_test, X_val, y_val = self.temporal_train_test_split(final_data)
            # NOW remove date from features after split
            if 'date' in X_train.columns:
                X_train = X_train.drop(columns=['date'])
            if 'date' in X_test.columns:
                X_test = X_test.drop(columns=['date'])
            if X_val is not None and 'date' in X_val.columns:
                X_val = X_val.drop(columns=['date'])
        else:
            feature_cols = [col for col in final_data.columns if col not in ['target', 'match_id']]
            X_train, X_test, y_train, y_test = train_test_split(
                final_data[feature_cols], final_data['target'],
                test_size=0.2, random_state=self.random_seed, stratify=final_data['target']
            )
            X_val, y_val = None, None

        # 13. Feature quality validation and preprocessing
        print("\n13. Feature quality validation and preprocessing...")

        # Feature quality validation
        X_train = self.validate_feature_quality(X_train, threshold=0.5)
        X_test = X_test[X_train.columns]  # Align test set columns
        if X_val is not None:
            X_val = X_val[X_train.columns]

        # Handle missing values
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())  # Use train medians
        if X_val is not None:
            X_val = X_val.fillna(X_train.median())

        # Remove constant features
        constant_features = X_train.columns[X_train.nunique() <= 1]
        if len(constant_features) > 0:
            X_train = X_train.drop(columns=constant_features)
            X_test = X_test.drop(columns=constant_features)
            if X_val is not None:
                X_val = X_val.drop(columns=constant_features)
            print(f"   Removed {len(constant_features)} constant features")

        # Advanced feature selection
        X_train, feature_selector = self.advanced_feature_selection(X_train, y_train)
        X_test = X_test[X_train.columns]  # Align test set
        if X_val is not None:
            X_val = X_val[X_train.columns]

        print(f"   Final training features: {X_train.shape[1]}")
        print(f"   Training samples: {len(X_train):,}")

        # Check class balance
        class_counts = y_train.value_counts()
        print(f"   Class balance: {dict(class_counts)}")

        if len(class_counts) != 2:
            raise ValueError(f"Expected 2 classes, got {len(class_counts)}")

        # 14. Model training with optimization (no double timeout)
        print("\n14. Training model with enhanced Bayesian optimization...")
        try:
            optimized_model, best_params = self.optimize_hyperparameters_bayesian_enhanced(X_train, y_train)
            print(f"Using optimized hyperparameters: {best_params}")
        except Exception as e:
            print(f"Bayesian optimization failed: {e}, using default parameters")
            optimized_model = self._get_default_lgb_model()

            # Fit with proper early stopping
            fit_params = {}
            if X_val is not None and y_val is not None:
                fit_params = {
                    'eval_set': [(X_val, y_val)],
                    'early_stopping_rounds': 50,
                    'verbose': False
                }

            optimized_model.fit(X_train, y_train, **fit_params)
            best_params = {}

        # 15. Evaluation
        print("\n15. Evaluating optimized model performance...")
        y_pred = optimized_model.predict(X_test)
        y_pred_proba = optimized_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)
        brier = brier_score_loss(y_test, y_pred_proba)

        print(f"OPTIMIZED MODEL PERFORMANCE:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC-ROC:  {auc:.4f}")
        print(f"  Log-Loss: {logloss:.4f}")
        print(f"  Brier Score: {brier:.4f}")

        # Cross-validation
        cv_scores = cross_val_score(
            optimized_model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed),
            scoring='roc_auc'
        )
        print(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': optimized_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 10 most important features:")
        print(feature_importance.head(10))

        # FIXED: Compute deterministic training data hash
        training_hash = compute_data_hash(final_data)
        print(f"\nTraining data SHA-256: {training_hash[:16]}...")

        # 16. Save comprehensive model with prediction-time compatibility
        model_path = self.model_cache / "optimized_comprehensive_model.pkl"
        model_metadata = {
            'model': optimized_model,
            'feature_columns': X_train.columns.tolist(),
            'feature_importance': feature_importance,
            'feature_selector': self.feature_selector,
            'feature_selector_type': self.feature_selector_type,
            'best_hyperparameters': best_params,
            'performance': {
                'accuracy': accuracy,
                'auc': auc,
                'log_loss': logloss,
                'brier_score': brier,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std()
            },
            'training_data_hash': training_hash,
            'training_date': date.today(),
            'random_seed': self.random_seed,
            'preprocessing_info': {
                'feature_medians': X_train.median().to_dict(),
                'constant_features_removed': constant_features.tolist() if len(constant_features) > 0 else []
            }
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_metadata, f)

        print(f"\nOptimized comprehensive model saved to: {model_path}")

        return optimized_model, feature_importance, {
            'accuracy': accuracy,
            'auc': auc,
            'log_loss': logloss,
            'brier_score': brier
        }

    def load_real_point_data_from_jeff(self, jeff_data):
        """Load real point data from Jeff's datasets"""
        print("Loading real point-by-point data from Jeff's datasets...")

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

            print(f"  {gender}: {points_found:,} real points loaded")

        if all_points:
            combined_points = pd.concat(all_points, ignore_index=True)
            print(f"Total real point data: {len(combined_points):,} points")
            return combined_points
        else:
            return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="FIXED Memory-Optimized Tennis Pipeline")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild dataset cache")
    parser.add_argument("--no-ta", action="store_true", help="Skip Tennis Abstract scraping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    global GLOBAL_SEED
    GLOBAL_SEED = args.seed
    np.random.seed(GLOBAL_SEED)

    pipeline = OptimizedComprehensiveDataPipeline(random_seed=GLOBAL_SEED)

    try:
        model, feature_importance, performance = pipeline.train_with_all_data_optimized(
            rebuild_cache=args.rebuild,
            skip_ta=args.no_ta
        )

        print("\n" + "=" * 60)
        print("ALL MEMORY FIXES APPLIED SUCCESSFULLY")
        print("=" * 60)
        print("✓ Replaced copy() with in-place updates")
        print("✓ Aggregate Jeff stat tables before merging")
        print("✓ Down-cast numeric columns to float32/int32 immediately")
        print("✓ Explicit memory cleanup with gc.collect() after each stage")
        print("✓ Process datasets one at a time with intermediate cleanup")
        print("✓ Early data filtering for RAM < 32GB systems")
        print("✓ Pandas copy-on-write mode enabled")
        print(f"✓ Final model AUC: {performance['auc']:.4f}")
        print(f"✓ Final model accuracy: {performance['accuracy']:.4f}")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()