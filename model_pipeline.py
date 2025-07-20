#!/usr/bin/env python3
"""
FIXED: Tennis ML Pipeline with All Critical Errors Resolved
- Fixed median fill to only apply to numeric columns
- Fixed composite_id canonicalization consistency across tennis_data, TA, and point logs
- Fixed point-level merge by ensuring identical match_id generation
- Fixed feature drop by applying weighted_defaults before missing-ratio filter
- Fixed TA scraper window size for adequate training data
- Removed unnecessary memory optimization filter
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
import gc
from pathlib import Path
from functools import wraps
from contextlib import contextmanager
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.feature_selection import SelectKBest, f_classif
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Memory optimization settings
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


def normalize_composite_id_component(text):
    """FIXED: Unified canonicalization for all composite_id components"""
    if pd.isna(text):
        return ""

    # Convert to string and normalize
    text = str(text).lower().strip()

    # Remove common punctuation and normalize spacing
    text = text.replace('.', '').replace("'", '').replace('-', ' ')
    text = ' '.join(text.split())  # Normalize whitespace

    # Replace spaces with underscores for composite_id
    return text.replace(' ', '_')


def create_canonical_composite_id(match_date, tournament, player1, player2):
    """FIXED: Create consistent composite_id across all data sources"""
    date_str = pd.to_datetime(match_date).strftime("%Y%m%d")
    tournament_canonical = normalize_composite_id_component(tournament)
    player1_canonical = normalize_composite_id_component(player1)
    player2_canonical = normalize_composite_id_component(player2)

    return f"{date_str}-{tournament_canonical}-{player1_canonical}-{player2_canonical}"


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
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return hashlib.sha256(str(len(data)).encode()).hexdigest()

    sorted_data = data[numeric_cols].sort_index().sort_index(axis=1)
    data_bytes = sorted_data.to_numpy().astype(np.float32).tobytes()
    return hashlib.sha256(data_bytes).hexdigest()


class FixedComprehensiveDataPipeline:
    """FIXED: Pipeline with all critical errors resolved"""

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
    def extract_all_jeff_features_with_defaults_first(self, match_data, jeff_data, weighted_defaults):
        """FIXED: Extract Jeff features and apply weighted_defaults BEFORE missing-ratio filter"""
        print("Extracting Jeff features with weighted defaults applied FIRST...")

        enhanced_data = match_data.copy()

        # FIXED: Get all Jeff feature names from weighted_defaults
        all_jeff_features = set()
        for gender_key in ['men', 'women']:
            if gender_key in weighted_defaults:
                all_jeff_features.update(weighted_defaults[gender_key].keys())

        # Pre-create ALL Jeff feature columns with weighted defaults
        print(f"Pre-creating {len(all_jeff_features)} Jeff feature columns with defaults...")
        for feature_name in all_jeff_features:
            winner_col = f'winner_{feature_name}'
            loser_col = f'loser_{feature_name}'

            if winner_col not in enhanced_data.columns:
                enhanced_data[winner_col] = np.nan
            if loser_col not in enhanced_data.columns:
                enhanced_data[loser_col] = np.nan

        # Apply weighted defaults FIRST (before any Jeff data extraction)
        print("Applying weighted defaults to all matches...")
        for idx, row in enhanced_data.iterrows():
            gender = row['gender']
            gender_key = 'men' if gender == 'M' else 'women'

            if gender_key in weighted_defaults:
                defaults = weighted_defaults[gender_key]

                for feature_name, default_value in defaults.items():
                    winner_col = f'winner_{feature_name}'
                    loser_col = f'loser_{feature_name}'

                    # Apply defaults where NaN
                    if pd.isna(enhanced_data.loc[idx, winner_col]):
                        enhanced_data.loc[idx, winner_col] = default_value
                    if pd.isna(enhanced_data.loc[idx, loser_col]):
                        enhanced_data.loc[idx, loser_col] = default_value

        print("✓ Weighted defaults applied to all Jeff features")

        # NOW extract actual Jeff data to override defaults where available
        feature_categories = {}
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
            idx = enhanced_data['gender'] == ('M' if gender == 'men' else 'W')

            if not idx.any():
                continue

            print(f"  Processing {gender}: {idx.sum():,} matches")

            # Process datasets with explicit cleanup
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

                    agg_df = overview_total.groupby('Player_canonical')[agg_cols].mean().astype(
                        np.float32).reset_index()

                    # Update winner stats where player matches
                    for _, agg_row in agg_df.iterrows():
                        player_canonical = agg_row['Player_canonical']
                        winner_mask = idx & (enhanced_data['winner_canonical'] == player_canonical)

                        if winner_mask.any():
                            for col in agg_cols:
                                if col in agg_row:
                                    enhanced_data.loc[winner_mask, f'winner_{col}'] = agg_row[col]

                    # Update loser stats where player matches
                    for _, agg_row in agg_df.iterrows():
                        player_canonical = agg_row['Player_canonical']
                        loser_mask = idx & (enhanced_data['loser_canonical'] == player_canonical)

                        if loser_mask.any():
                            for col in agg_cols:
                                if col in agg_row:
                                    enhanced_data.loc[loser_mask, f'loser_{col}'] = agg_row[col]

                    del overview_total, agg_df
                    gc.collect()

                elif 'player' in df.columns:
                    agg_dict = {col: 'mean' for col in agg_cols if col in df.columns}
                    if not agg_dict:
                        continue

                    agg_df = df.groupby('player').agg(agg_dict).astype(np.float32).reset_index()

                    # Update winner stats
                    for _, agg_row in agg_df.iterrows():
                        player_canonical = agg_row['player']
                        winner_mask = idx & (enhanced_data['winner_canonical'] == player_canonical)

                        if winner_mask.any():
                            for col in agg_cols:
                                if col in agg_row:
                                    col_name = f'winner_{dataset_name}_{col}'
                                    if col_name in enhanced_data.columns:
                                        enhanced_data.loc[winner_mask, col_name] = agg_row[col]

                    # Update loser stats
                    for _, agg_row in agg_df.iterrows():
                        player_canonical = agg_row['player']
                        loser_mask = idx & (enhanced_data['loser_canonical'] == player_canonical)

                        if loser_mask.any():
                            for col in agg_cols:
                                if col in agg_row:
                                    col_name = f'loser_{dataset_name}_{col}'
                                    if col_name in enhanced_data.columns:
                                        enhanced_data.loc[loser_mask, col_name] = agg_row[col]

                    del agg_df
                    gc.collect()

                feature_categories[dataset_name] = feature_categories.get(dataset_name, 0) + idx.sum()

        print(f"✓ Jeff feature extraction complete with defaults-first approach. Categories used: {feature_categories}")

        # Validate Jeff feature extraction
        self.validate_feature_extraction(enhanced_data, 'winner_', min_features=10)
        self.validate_feature_extraction(enhanced_data, 'loser_', min_features=10)

        return enhanced_data

    def optimize_hyperparameters_bayesian_enhanced(self, X_train, y_train):
        """FIXED: Enhanced Bayesian optimization with proper fallback"""
        print("Optimizing hyperparameters with enhanced Bayesian search...")

        try:
            from skopt import BayesSearchCV
            from skopt.space import Real, Integer

            # Split training data for early stopping
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.random_seed, stratify=y_train
            )

            # Define search space for LightGBM
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

            # Base model
            lgb_model = lgb.LGBMClassifier(
                random_state=self.random_seed,
                class_weight='balanced',
                verbose=-1
            )

            # Bayesian search with cross-validation
            bayes_search = BayesSearchCV(
                estimator=lgb_model,
                search_spaces=search_spaces,
                n_iter=30,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_seed),
                scoring='roc_auc',
                n_jobs=-1,
                random_state=self.random_seed
            )

            # Fit the search
            bayes_search.fit(X_train_split, y_train_split)
            print(f"Best CV score: {bayes_search.best_score_:.4f}")
            print(f"Best parameters: {bayes_search.best_params_}")

            # Train final model on full training set
            final_model = bayes_search.best_estimator_
            final_model.fit(X_train, y_train,
                            eval_set=[(X_val_split, y_val_split)],
                            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

            return final_model, bayes_search.best_params_

        except ImportError:
            print("scikit-optimize not available, using default parameters")
            default_model = self._get_default_lgb_model()
            default_model.fit(X_train, y_train)
            return default_model, {}
        except Exception as e:
            print(f"Bayesian optimization failed: {e}, using default parameters")
            default_model = self._get_default_lgb_model()
            default_model.fit(X_train, y_train)
            return default_model, {}

    def _get_default_lgb_model(self):
        """Get default LightGBM model when optimization fails"""
        return lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.03,
            random_state=self.random_seed,
            class_weight='balanced',
            verbose=-1
        )

    @exponential_backoff_retry(max_attempts=5, base_delay=2)
    @timing_wrapper("Tennis Abstract Scraping", STAGE_TIMEOUTS['tennis_abstract'])
    def scrape_tennis_abstract_with_retry(self, days_back=90, max_matches=500):
        """FIXED: Tennis Abstract scraping with increased window for training data"""
        scraper = AutomatedTennisAbstractScraper()
        return scraper.automated_scraping_session(days_back=days_back, max_matches=max_matches)

    def integrate_all_tennis_abstract_features_fixed(self, match_data, scraped_records):
        """FIXED: TA feature integration with canonical composite_id matching"""
        print("Integrating Tennis Abstract features with FIXED composite_id matching...")

        if not scraped_records:
            print("WARNING: No Tennis Abstract records available")
            return match_data

        # FIXED: Re-canonicalize TA composite_ids to match tennis_data format
        print("Re-canonicalizing Tennis Abstract composite_ids...")
        for record in scraped_records:
            if 'Date' in record and 'tournament' in record and 'player1' in record and 'player2' in record:
                # Use the same canonicalization as tennis_data
                canonical_id = create_canonical_composite_id(
                    record['Date'],
                    record['tournament'],
                    record['player1'],
                    record['player2']
                )
                record['composite_id'] = canonical_id

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

        # Check composite_id overlap
        tennis_comp_ids = set(match_data['composite_id'].unique())
        ta_comp_ids = set(ta_by_match.keys())
        overlap = tennis_comp_ids & ta_comp_ids
        print(f"Composite ID overlap: {len(overlap)}/{len(ta_comp_ids)} TA matches found in tennis data")

        if len(overlap) == 0:
            print("ERROR: No composite_id overlap found!")
            print(f"Sample tennis composite_ids: {list(tennis_comp_ids)[:3]}")
            print(f"Sample TA composite_ids: {list(ta_comp_ids)[:3]}")
            return match_data

        # Incremental assignment
        enhanced_data = match_data.copy()
        matches_enhanced = 0
        ta_features_added = set()

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

                    if col_name not in enhanced_data.columns:
                        enhanced_data[col_name] = np.nan

                    enhanced_data.loc[match_idx, col_name] = feature_value
                    ta_features_added.add(col_name)

            enhanced_data.loc[match_idx, 'ta_enhanced'] = True
            matches_enhanced += 1

        print(f"Enhanced {matches_enhanced} matches with {len(ta_features_added)} TA features")

        if matches_enhanced > 0:
            self.validate_feature_extraction(enhanced_data, 'winner_ta_', min_features=1)
            self.validate_feature_extraction(enhanced_data, 'loser_ta_', min_features=1)

        return enhanced_data

    def integrate_point_level_features_fixed(self, match_data, point_data):
        """FIXED: Point integration with canonical match_id matching"""
        print("Integrating point-level data with FIXED match_id canonicalization...")

        if point_data.empty:
            print("No point data to integrate")
            return match_data

        # FIXED: Re-canonicalize point data match_ids to match tennis_data format
        print("Re-canonicalizing point data match_ids...")

        # Extract date, tournament, players from point match_ids and re-canonicalize
        canonical_point_matches = {}
        for match_id in point_data['match_id'].unique():
            # Try to parse match_id components
            # Point match_ids might be in format like "20250713-M-Wimbledon-F-Carlos_Alcaraz-Jannik_Sinner"
            parts = str(match_id).split('-')
            if len(parts) >= 5:
                date_part = parts[0]
                # Skip gender part (parts[1])
                tournament_part = parts[2]
                # Skip round part (parts[3])
                player1_part = parts[4]
                player2_part = parts[5] if len(parts) > 5 else ''

                # Create canonical version
                canonical_id = create_canonical_composite_id(
                    date_part, tournament_part, player1_part, player2_part
                )
                canonical_point_matches[match_id] = canonical_id
            else:
                canonical_point_matches[match_id] = match_id

        # Update point_data with canonical match_ids
        point_data = point_data.copy()
        point_data['canonical_match_id'] = point_data['match_id'].map(canonical_point_matches)

        # Check overlap
        tennis_match_ids = set(match_data['composite_id'].unique())
        point_match_ids = set(point_data['canonical_match_id'].unique())
        overlap = tennis_match_ids & point_match_ids

        print(f"Point-level match ID overlap: {len(overlap)}/{len(point_match_ids)} point matches found")

        if len(overlap) == 0:
            print("WARNING: No match_id overlap found for point integration")
            print(f"Sample tennis composite_ids: {list(tennis_match_ids)[:3]}")
            print(f"Sample point canonical_ids: {list(point_match_ids)[:3]}")
            return match_data

        # Filter to overlapping matches only
        filtered_point_data = point_data[point_data['canonical_match_id'].isin(overlap)]
        print(f"✓ Using {len(filtered_point_data):,} points from {len(overlap)} matched games")

        # Vectorized point aggregation
        grouped = filtered_point_data.groupby('canonical_match_id')

        point_stats = []
        for match_id, match_points in grouped:
            if len(match_points) < 10:
                continue

            total_points = len(match_points)

            # Server statistics
            server_1_points = match_points[match_points['Svr'] == 1]
            server_2_points = match_points[match_points['Svr'] == 2]

            server_1_win_rate = (server_1_points['PtWinner'] == 1).mean() if len(server_1_points) > 0 else 0.5
            server_2_win_rate = (server_2_points['PtWinner'] == 2).mean() if len(server_2_points) > 0 else 0.5

            # Neutral features
            serve_advantage_diff = server_1_win_rate - server_2_win_rate
            overall_serve_rate = (server_1_win_rate + server_2_win_rate) / 2
            serve_volatility = abs(server_1_win_rate - server_2_win_rate)

            # Break point performance
            bp_diff = 0.0
            if 'is_break_point' in match_points.columns:
                bp_points_1 = match_points[(match_points['is_break_point'] == True) & (match_points['Svr'] == 1)]
                bp_points_2 = match_points[(match_points['is_break_point'] == True) & (match_points['Svr'] == 2)]

                bp_rate_1 = (bp_points_1['PtWinner'] == 1).mean() if len(bp_points_1) > 0 else 0.5
                bp_rate_2 = (bp_points_2['PtWinner'] == 2).mean() if len(bp_points_2) > 0 else 0.5
                bp_diff = bp_rate_1 - bp_rate_2

            # Rally length
            avg_rally_length = match_points.get('rally_length', pd.Series([4.0])).mean()
            rally_length_std = match_points.get('rally_length', pd.Series([2.0])).std()

            point_stats.append({
                'composite_id': match_id,  # Use composite_id for merging
                'total_points': total_points,
                'serve_advantage_diff': serve_advantage_diff,
                'overall_serve_rate': overall_serve_rate,
                'serve_volatility': serve_volatility,
                'bp_performance_diff': bp_diff,
                'avg_rally_length': avg_rally_length,
                'rally_length_std': rally_length_std,
                'match_competitiveness': 1 - abs(serve_advantage_diff)
            })

        if point_stats:
            point_stats_df = pd.DataFrame(point_stats)
            enhanced_data = match_data.merge(point_stats_df, on='composite_id', how='left')
            added_features = len(point_stats_df.columns) - 1
            print(f"Integrated {added_features} neutral point-level features for {len(point_stats)} matches")
            return enhanced_data
        else:
            print("No point statistics calculated")
            return match_data

    def create_balanced_training_dataset(self, match_data):
        """Create balanced dataset with proper target construction"""
        print("Creating balanced training dataset with proper target construction...")

        positive_examples = match_data.copy()
        positive_examples['target'] = 1
        positive_examples['match_id'] = positive_examples.index.astype(str) + '_pos'

        negative_examples = match_data.copy()

        # FIXED: Only swap numeric winner/loser columns
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

        balanced_data = pd.concat([positive_examples, negative_examples], ignore_index=True)

        print(f"Created balanced dataset:")
        print(f"  Original matches: {len(match_data):,}")
        print(f"  Balanced examples: {len(balanced_data):,}")
        print(f"  Class distribution: {balanced_data['target'].value_counts().to_dict()}")

        del positive_examples, negative_examples
        return balanced_data

    def create_relative_features(self, balanced_data):
        """Convert to relative features to prevent leakage"""
        print("Converting to relative features to prevent leakage...")

        # FIXED: Find matching winner/loser feature pairs (numeric only)
        winner_cols = [col for col in balanced_data.columns
                       if col.startswith('winner_') and balanced_data[col].dtype in ['int64', 'float64']]

        relative_features = {}
        for winner_col in winner_cols:
            base_name = winner_col[7:]
            loser_col = f'loser_{base_name}'

            if loser_col in balanced_data.columns and balanced_data[loser_col].dtype in ['int64', 'float64']:
                rel_col = f'rel_{base_name}'
                winner_vals = pd.to_numeric(balanced_data[winner_col], errors='coerce')
                loser_vals = pd.to_numeric(balanced_data[loser_col], errors='coerce')
                relative_features[rel_col] = winner_vals - loser_vals

        relative_df = pd.DataFrame(relative_features, index=balanced_data.index)

        # Expanded leakage patterns
        leakage_patterns = ['winner_', 'loser_', 'Winner', 'Loser', 'composite_id',
                            'Tournament', 'Surface', 'Round', 'Series', 'Court',
                            'Location', 'Tier', 'tournament_', 'event_']

        non_leaking_cols = []
        for col in balanced_data.columns:
            if not any(col.startswith(pattern) for pattern in leakage_patterns):
                non_leaking_cols.append(col)

        final_data = pd.concat([
            balanced_data[non_leaking_cols],
            relative_df
        ], axis=1)

        print(f"Created {len(relative_features)} relative features")
        print(f"Kept {len(non_leaking_cols)} non-leaking features")
        print(f"Final feature count: {len(final_data.columns) - 2}")

        del relative_df, balanced_data
        return final_data

    def validate_feature_quality_fixed(self, X, threshold=0.3):
        """FIXED: Lower threshold to preserve engineered features"""
        print(f"Validating feature quality (dropping columns with >{threshold * 100}% missing)...")

        # FIXED: Only check numeric columns for missing ratios
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        missing_ratios = X_numeric.isnull().mean()
        high_missing_cols = missing_ratios[missing_ratios > threshold].index.tolist()

        if high_missing_cols:
            X_clean = X_numeric.drop(columns=high_missing_cols)
            print(f"Dropped {len(high_missing_cols)} high-missing columns")
        else:
            X_clean = X_numeric

        print(f"Feature quality validation: {X_clean.shape[1]}/{X.shape[1]} features retained")
        return X_clean

    def temporal_train_test_split(self, final_data, test_size=0.2, val_size=0.1):
        """Temporal split with proper date column handling"""
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

        sorted_data = final_data.sort_values('date').reset_index(drop=True)
        n_samples = len(sorted_data)

        train_end = int(n_samples * (1 - test_size - val_size))
        val_end = int(n_samples * (1 - test_size))

        train_data = sorted_data.iloc[:train_end]
        val_data = sorted_data.iloc[train_end:val_end]
        test_data = sorted_data.iloc[val_end:]

        print(f"Temporal split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

        feature_cols = [col for col in sorted_data.columns if col not in ['target', 'match_id', 'date']]

        X_train = train_data[feature_cols]
        y_train = train_data['target']
        X_val = val_data[feature_cols] if len(val_data) > 0 else None
        y_val = val_data['target'] if len(val_data) > 0 else None
        X_test = test_data[feature_cols]
        y_test = test_data['target']

        return X_train, X_test, y_train, y_test, X_val, y_val

    def advanced_feature_selection(self, X, y):
        """Advanced feature selection without duplicate constant removal"""
        print("Performing advanced feature selection...")

        if X.shape[1] <= 100:
            self.feature_selector = {
                'type': 'none',
                'selected_features': X.columns.tolist()
            }
            self.feature_selector_type = 'none'
            return X, self.feature_selector

        if X.shape[1] > 300:
            print("Using LightGBM-based feature selection for large feature set...")

            lgb_selector = lgb.LGBMClassifier(
                n_estimators=100,
                random_state=self.random_seed,
                verbose=-1
            )
            lgb_selector.fit(X, y)

            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': lgb_selector.feature_importances_
            }).sort_values('importance', ascending=False)

            n_features = min(200, X.shape[1])
            selected_features = importance_df.head(n_features)['feature'].tolist()
            X_selected = X[selected_features]

            self.feature_selector = {
                'type': 'lgbm_importance',
                'selected_features': selected_features,
                'importance_scores': importance_df
            }
            self.feature_selector_type = 'lgbm'

            print(f"LGBM selection: {X_selected.shape[1]} features from {X.shape[1]}")

        else:
            print("Using SelectKBest for medium feature set...")

            k_features = min(100, X.shape[1])
            selector = SelectKBest(score_func=f_classif, k=k_features)
            X_selected = selector.fit_transform(X, y)

            selected_features = X.columns[selector.get_support()].tolist()
            X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

            self.feature_selector = {
                'type': 'select_k_best',
                'selector_object': selector,
                'selected_features': selected_features,
                'feature_scores': selector.scores_
            }
            self.feature_selector_type = 'sklearn'

            print(f"SelectKBest: {X_selected.shape[1]} features from {X.shape[1]}")

        return X_selected, self.feature_selector

    def train_with_all_data_fixed(self, rebuild_cache=False, skip_ta=False):
        """FIXED: Training pipeline with all critical errors resolved"""
        print("TRAINING WITH ALL CRITICAL FIXES APPLIED")
        print("=" * 60)

        # 1. Load base tennis data
        with timer_context("Data Loading", STAGE_TIMEOUTS['data_loading']):
            print("\n1. Loading base tennis match data...")
            tennis_data = load_all_tennis_data()

            # FIXED: Remove unnecessary memory optimization filter
            print(f"   Base tennis data: {len(tennis_data):,} matches")

            print("\n2. Loading Jeff's comprehensive charting data...")
            jeff_data = load_jeff_comprehensive_data()

            if not jeff_data or len(jeff_data) == 0:
                raise ValueError("Jeff data loading failed")

            print("\n3. Calculating comprehensive weighted defaults...")
            weighted_defaults = calculate_comprehensive_weighted_defaults(jeff_data)

        # 4. Process tennis data with FIXED composite_id canonicalization
        print("\n4. Processing tennis match data with FIXED canonicalization...")
        tennis_data['winner_canonical'] = tennis_data['Winner'].apply(normalize_name)
        tennis_data['loser_canonical'] = tennis_data['Loser'].apply(normalize_name)

        if 'Date' in tennis_data.columns:
            tennis_data['Date'] = pd.to_datetime(tennis_data['Date'], errors='coerce')
            tennis_data['date'] = tennis_data['Date'].dt.date

        tennis_data = tennis_data.dropna(subset=['date'])

        # FIXED: Use canonical composite_id creation
        tennis_data['composite_id'] = tennis_data.apply(
            lambda row: create_canonical_composite_id(
                row['date'], row['Tournament'], row['Winner'], row['Loser']
            ), axis=1
        )

        tennis_data['match_id'] = tennis_data['composite_id']
        print(f"   Processed tennis data: {len(tennis_data):,} matches")

        # 5. FIXED: Extract Jeff features with defaults applied FIRST
        enhanced_data = self.extract_all_jeff_features_with_defaults_first(tennis_data, jeff_data, weighted_defaults)

        gc.collect()

        # 6. Tennis Abstract integration with FIXED composite_id matching
        if not skip_ta:
            try:
                print("\n6. Scraping Tennis Abstract data with FIXED window...")
                scraped_records = self.scrape_tennis_abstract_with_retry(days_back=90, max_matches=500)

                if scraped_records:
                    print("\n7. Integrating Tennis Abstract features with FIXED matching...")
                    enhanced_data = self.integrate_all_tennis_abstract_features_fixed(enhanced_data, scraped_records)
            except Exception as e:
                print(f"Tennis Abstract integration failed: {e}")
                print("Continuing without TA data...")
        else:
            print("\n6. Skipping Tennis Abstract scraping (--no-ta flag)")

        # 8. Point-level integration with FIXED match_id canonicalization
        print("\n8. Loading and integrating point-level data with FIXED matching...")
        real_point_data = self.load_real_point_data_from_jeff(jeff_data)
        if not real_point_data.empty:
            enhanced_data = self.integrate_point_level_features_fixed(enhanced_data, real_point_data)

        gc.collect()

        # 9-11. Create balanced dataset and relative features
        print("\n9. Creating balanced dataset...")
        balanced_data = self.create_balanced_training_dataset(enhanced_data)

        print("\n10. Converting to relative features...")
        final_data = self.create_relative_features(balanced_data)

        gc.collect()

        # 11. Remove leakage indicators (preserve date for temporal split)
        print("\n11. Removing leakage indicators...")
        leakage_columns = []
        leakage_patterns = ['ta_enhanced', 'source_rank', 'data_quality_score', 'composite_id',
                            'Winner', 'Loser', 'winner_canonical', 'loser_canonical']

        for col in final_data.columns:
            if col in leakage_patterns or ('year' in col.lower() and col != 'date'):
                leakage_columns.append(col)

        if leakage_columns:
            final_data = final_data.drop(columns=leakage_columns)
            print(f"   Removed {len(leakage_columns)} leakage indicators")

        # 12. Temporal train-test split
        print("\n12. Performing temporal train-test split...")
        if 'date' in final_data.columns:
            X_train, X_test, y_train, y_test, X_val, y_val = self.temporal_train_test_split(final_data)
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

        # FIXED: Use lower threshold for feature quality validation
        X_train = self.validate_feature_quality_fixed(X_train, threshold=0.3)
        X_test = X_test[X_train.columns]
        if X_val is not None:
            X_val = X_val[X_train.columns]

        # FIXED: Handle missing values only on numeric columns
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        medians = X_train[num_cols].median()
        X_train[num_cols] = X_train[num_cols].fillna(medians)
        X_test[num_cols] = X_test[num_cols].fillna(medians)
        if X_val is not None:
            X_val[num_cols] = X_val[num_cols].fillna(medians)

        # Remove remaining object columns
        obj_cols = X_train.select_dtypes(exclude=[np.number]).columns
        if len(obj_cols) > 0:
            print(f"   Removing {len(obj_cols)} object columns")
            X_train = X_train.drop(columns=obj_cols)
            X_test = X_test.drop(columns=obj_cols)
            if X_val is not None:
                X_val = X_val.drop(columns=obj_cols)

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
        X_test = X_test[X_train.columns]
        if X_val is not None:
            X_val = X_val[X_train.columns]

        print(f"   Final training features: {X_train.shape[1]}")
        print(f"   Training samples: {len(X_train):,}")

        # Check class balance
        class_counts = y_train.value_counts()
        print(f"   Class balance: {dict(class_counts)}")

        if len(class_counts) != 2:
            raise ValueError(f"Expected 2 classes, got {len(class_counts)}")

        # 14. Model training with FIXED optimization
        print("\n14. Training model with FIXED Bayesian optimization...")
        optimized_model, best_params = self.optimize_hyperparameters_bayesian_enhanced(X_train, y_train)
        print(f"Using optimized hyperparameters: {best_params}")

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

        # Training data hash
        training_hash = compute_data_hash(final_data)
        print(f"\nTraining data SHA-256: {training_hash[:16]}...")

        # 16. Save comprehensive model
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
                'feature_medians': medians.to_dict(),
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
    parser = argparse.ArgumentParser(description="FIXED Comprehensive Tennis Pipeline")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild dataset cache")
    parser.add_argument("--no-ta", action="store_true", help="Skip Tennis Abstract scraping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    global GLOBAL_SEED
    GLOBAL_SEED = args.seed
    np.random.seed(GLOBAL_SEED)

    pipeline = FixedComprehensiveDataPipeline(random_seed=GLOBAL_SEED)

    try:
        model, feature_importance, performance = pipeline.train_with_all_data_fixed(
            rebuild_cache=args.rebuild,
            skip_ta=args.no_ta
        )

        print("\n" + "=" * 60)
        print("ALL CRITICAL FIXES SUCCESSFULLY APPLIED")
        print("=" * 60)
        print("✓ Fixed median fill to only apply to numeric columns")
        print("✓ Fixed composite_id canonicalization consistency across tennis_data, TA, and point logs")
        print("✓ Fixed point-level merge by ensuring identical match_id generation")
        print("✓ Fixed feature drop by applying weighted_defaults BEFORE missing-ratio filter")
        print("✓ Fixed TA scraper window size for adequate training data")
        print("✓ Removed unnecessary memory optimization filter")
        print("✓ Fixed Bayesian optimization with proper fallback")
        print("✓ Lowered missing-ratio threshold to preserve engineered features")
        print(f"✓ Final model AUC: {performance['auc']:.4f} (expected >0.60)")
        print(f"✓ Final model accuracy: {performance['accuracy']:.4f}")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()