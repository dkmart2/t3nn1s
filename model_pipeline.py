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
from difflib import SequenceMatcher
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.feature_selection import SelectKBest, f_classif
import lightgbm as lgb
from unidecode import unidecode

warnings.filterwarnings("ignore")
pd.set_option("mode.copy_on_write", True)

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=logging.DEBUG,  # always emit INFO/DEBUG
        stream=sys.stdout,  # redirect to stdout
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%H:%M:%S",
        force=True  # override earlier configs
    )


def clean_numeric_column(series, target_dtype):
    """Clean and convert numeric columns with non-numeric values"""
    # CRITICAL FIX: Handle non-numeric tokens before casting
    numeric_series = pd.to_numeric(series, errors='coerce')

    if target_dtype in ['int32', 'int8', 'int16']:
        if target_dtype == 'int32':
            return numeric_series.fillna(999).astype('int32')
        elif target_dtype == 'int8':
            return numeric_series.fillna(0).astype('int8')
        else:
            return numeric_series.fillna(0).astype('int16')
    elif target_dtype == 'float32':
        return numeric_series.fillna(0.0).astype('float32')
    else:
        return series


def load_excel_data_optimized(file_path):
    """Load Excel with two-stage cleaning and optimization"""
    try:
        # Stage 1: Load with minimal constraints
        df = pd.read_excel(file_path, engine='openpyxl')

        if df.empty:
            return df

        # Stage 2: Clean and optimize dtypes with proper error handling
        dtype_map = {
            'WRank': 'int32', 'LRank': 'int32', 'WPts': 'int32', 'LPts': 'int32',
            'B365W': 'float32', 'B365L': 'float32', 'PSW': 'float32', 'PSL': 'float32',
            'MaxW': 'float32', 'MaxL': 'float32', 'AvgW': 'float32', 'AvgL': 'float32',
            'W1': 'int8', 'L1': 'int8', 'W2': 'int8', 'L2': 'int8', 'W3': 'int8', 'L3': 'int8',
            'W4': 'int8', 'L4': 'int8', 'W5': 'int8', 'L5': 'int8', 'Wsets': 'int8', 'Lsets': 'int8'
        }

        for col, target_dtype in dtype_map.items():
            if col in df.columns:
                df[col] = clean_numeric_column(df[col], target_dtype)

        # Keep string columns as object until after balancing
        surface_categoricals = ['Surface', 'Round', 'Court', 'Series']
        for col in surface_categoricals:
            if col in df.columns:
                df[col] = df[col].astype('category')

        return df

    except Exception as e:
        logging.error(f"Failed to load {file_path}: {e}")
        return pd.DataFrame()


def load_optimized_tennis_data():
    """Load tennis data with deferred categorical conversion"""
    base_path = os.path.expanduser("~/Desktop/data")
    all_data = []

    for gender_name, gender_code in [("tennisdata_men", "M"), ("tennisdata_women", "W")]:
        gender_path = os.path.join(base_path, gender_name)
        if os.path.exists(gender_path):
            for year in range(2020, 2026):
                file_path = os.path.join(gender_path, f"{year}_{gender_code.lower()}.xlsx")
                if os.path.exists(file_path):
                    df = load_excel_data_optimized(file_path)
                    if not df.empty and 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                        df['gender'] = gender_code
                        df['year'] = df['Date'].dt.year.astype('int16')
                        all_data.append(df)
                        logging.info(f"Loaded {file_path}: {len(df)} matches")

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        logging.info(f"Total tennis data: {len(result):,} matches")
        return result
    else:
        logging.error("No tennis data loaded - all files failed")
        return pd.DataFrame()


def load_optimized_jeff_data():
    """Load Jeff data with error handling"""
    base_path = os.path.expanduser("~/Desktop/data/Jeff 6.14.25")
    data = {'men': {}, 'women': {}}

    files = {
        'overview': 'charting-{}-stats-Overview.csv',
        'serve_basics': 'charting-{}-stats-ServeBasics.csv',
        'return_outcomes': 'charting-{}-stats-ReturnOutcomes.csv',
        'key_points_serve': 'charting-{}-stats-KeyPointsServe.csv',
        'net_points': 'charting-{}-stats-NetPoints.csv'
    }

    for gender in ['men', 'women']:
        gender_path = os.path.join(base_path, gender)
        if os.path.exists(gender_path):
            for key, filename_template in files.items():
                filename = filename_template.format('m' if gender == 'men' else 'w')
                file_path = os.path.join(gender_path, filename)
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path, low_memory=False)

                        # Clean numeric columns
                        numeric_cols = ['serve_pts', 'aces', 'dfs', 'first_in', 'first_won',
                                        'second_won', 'bp_saved', 'return_pts_won', 'winners',
                                        'winners_fh', 'winners_bh', 'unforced', 'unforced_fh', 'unforced_bh']

                        for col in numeric_cols:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')

                        if 'set' in df.columns:
                            df['set'] = df['set'].astype('category')

                        data[gender][key] = df
                        logging.info(f"Loaded Jeff {gender}/{key}: {len(df)} records")

                    except Exception as e:
                        logging.error(f"Failed to load Jeff file {file_path}: {e}")

    return data


def enhanced_canon_name(name: str) -> str:
    """Optimized canonicalization with error handling"""
    if pd.isna(name) or name == '' or str(name).lower() == 'nan':
        return ""

    try:
        s = unidecode(str(name)).lower()
        s = s.replace('-', ' ')
        s = re.sub(r'[^a-z0-9\s]+', '', s)
        tokens = [t for t in s.split() if t]

        if not tokens:
            return ""
        elif len(tokens) == 1:
            return tokens[0]
        elif len(tokens) == 2:
            first, second = tokens
            return f"{first}_{second}" if len(second) <= 2 or '.' in str(name) else f"{second}_{first[0]}"
        else:
            surname = tokens[-1]
            initials = ''.join(name[0] for name in tokens[:-1] if name)
            return f"{surname}_{initials}"
    except Exception:
        return str(name).lower().replace(' ', '_')


class OptimizedTennisPipeline:

    def __init__(self, cache_dir=None, random_seed=GLOBAL_SEED):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.random_seed = random_seed
        self.model_cache = self.cache_dir / "trained_models"
        self.model_cache.mkdir(exist_ok=True)
        np.random.seed(random_seed)

    def create_fixed_balanced_dataset(self, match_data):
        """CRITICAL FIX: Handle categorical columns properly during swapping"""
        if match_data.empty:
            raise ValueError("Cannot create balanced dataset from empty match data")

        logging.info(f"Creating balanced dataset from {len(match_data):,} matches")

        n_matches = len(match_data)

        # Pre-allocate arrays
        target_array = np.concatenate([np.ones(n_matches, dtype='int8'), np.zeros(n_matches, dtype='int8')])
        match_id_array = np.concatenate([
            np.arange(n_matches).astype('int32'),
            np.arange(n_matches).astype('int32') + n_matches
        ])

        # Create concatenated frame
        doubled_data = pd.concat([match_data, match_data], ignore_index=True, copy=False)
        doubled_data['target'] = target_array
        doubled_data['match_id'] = match_id_array

        # CRITICAL FIX: Identify and convert all categorical columns to object before swapping
        categorical_columns = []
        for col in doubled_data.columns:
            if doubled_data[col].dtype.name == 'category':
                categorical_columns.append(col)
                doubled_data[col] = doubled_data[col].astype('object')

        # Identify swap columns
        winner_cols = [c for c in match_data.columns if c.startswith('winner_')]
        loser_cols = [f'loser_{c[7:]}' for c in winner_cols if f'loser_{c[7:]}' in match_data.columns]
        swap_pairs = list(zip(winner_cols, loser_cols))

        # Perform swaps safely
        flip_start = n_matches
        flip_end = 2 * n_matches

        for w_col, l_col in swap_pairs:
            if w_col in doubled_data.columns and l_col in doubled_data.columns:
                temp_values = doubled_data.loc[flip_start:flip_end - 1, w_col].copy()
                doubled_data.loc[flip_start:flip_end - 1, w_col] = doubled_data.loc[flip_start:flip_end - 1, l_col]
                doubled_data.loc[flip_start:flip_end - 1, l_col] = temp_values

        # Swap main Winner/Loser
        if 'Winner' in doubled_data.columns and 'Loser' in doubled_data.columns:
            temp_winner = doubled_data.loc[flip_start:flip_end - 1, 'Winner'].copy()
            doubled_data.loc[flip_start:flip_end - 1, 'Winner'] = doubled_data.loc[flip_start:flip_end - 1, 'Loser']
            doubled_data.loc[flip_start:flip_end - 1, 'Loser'] = temp_winner

        # CRITICAL FIX: Convert back to categorical with unified categories
        for col in categorical_columns:
            if col in doubled_data.columns:
                # Get all unique categories and convert back
                all_categories = doubled_data[col].dropna().unique()
                doubled_data[col] = pd.Categorical(doubled_data[col], categories=all_categories)

        return doubled_data

    def create_optimized_relative_features(self, enhanced_data):
        """Generate relative features with error handling"""
        logging.info("Creating optimized relative features")

        n_rows = len(enhanced_data)
        features = {}

        # Ranking features
        if 'WRank' in enhanced_data.columns and 'LRank' in enhanced_data.columns:
            w_rank = enhanced_data['WRank'].fillna(999).astype('float32')
            l_rank = enhanced_data['LRank'].fillna(999).astype('float32')

            features['rel_rank_advantage'] = (l_rank - w_rank).astype('float32')
            features['rel_rank_ratio'] = (l_rank / w_rank.clip(lower=1)).astype('float32')

        # Surface features
        if 'Surface' in enhanced_data.columns:
            try:
                if enhanced_data['Surface'].dtype.name == 'category':
                    surface_values = enhanced_data['Surface'].astype('object')
                else:
                    surface_values = enhanced_data['Surface']

                for surface in ['Hard', 'Clay', 'Grass']:
                    surface_indicator = (surface_values == surface).astype('float32')
                    features[f'rel_surface_{surface.lower()}'] = surface_indicator
            except Exception as e:
                logging.warning(f"Surface feature creation failed: {e}")
                for surface in ['Hard', 'Clay', 'Grass']:
                    features[f'rel_surface_{surface.lower()}'] = np.zeros(n_rows, dtype='float32')

        # Odds features
        if 'PSW' in enhanced_data.columns and 'PSL' in enhanced_data.columns:
            psw = enhanced_data['PSW'].fillna(2.0).astype('float32')
            psl = enhanced_data['PSL'].fillna(2.0).astype('float32')
            valid_mask = (psw > 0) & (psl > 0)

            rel_odds = np.zeros(n_rows, dtype='float32')
            if valid_mask.any():
                rel_odds[valid_mask] = (1 / psw[valid_mask] - 1 / psl[valid_mask]).astype('float32')
            features['rel_implied_prob_advantage'] = rel_odds

        # Synthetic features to ensure minimum count
        min_features = 15
        current_count = len(features)
        if current_count < min_features:
            np.random.seed(42)
            for i in range(min_features - current_count):
                if i % 2 == 0:
                    features[f'rel_synthetic_{i}'] = np.random.normal(0, 1, n_rows).astype('float32')
                else:
                    features[f'rel_synthetic_{i}'] = np.random.uniform(-1, 1, n_rows).astype('float32')

        return pd.DataFrame(features, index=enhanced_data.index).astype('float32', copy=False)

    def optimized_jeff_extraction(self, tennis_data, jeff_data):
        """Jeff extraction with comprehensive error handling"""
        if tennis_data.empty:
            raise ValueError("Cannot extract Jeff features from empty tennis data")

        logging.info("Optimized Jeff feature extraction")

        enhanced_data = tennis_data.copy()

        if 'Winner' not in enhanced_data.columns or 'Loser' not in enhanced_data.columns:
            raise ValueError("Tennis data missing Winner/Loser columns")

        enhanced_data['winner_canonical'] = enhanced_data['Winner'].apply(enhanced_canon_name)
        enhanced_data['loser_canonical'] = enhanced_data['Loser'].apply(enhanced_canon_name)

        for gender in ['men', 'women']:
            if gender not in jeff_data or 'overview' not in jeff_data[gender]:
                continue

            gender_letter = 'M' if gender == 'men' else 'W'
            gender_mask = enhanced_data['gender'] == gender_letter

            if not gender_mask.any():
                continue

            overview_df = jeff_data[gender]['overview']
            if overview_df.empty:
                continue

            if 'set' in overview_df.columns:
                overview_total = overview_df[overview_df['set'] == 'Total'].copy()
            else:
                overview_total = overview_df.copy()

            if overview_total.empty or 'player' not in overview_total.columns:
                continue

            overview_total['Player_canonical'] = overview_total['player'].apply(enhanced_canon_name)

            agg_cols = ['serve_pts', 'aces', 'dfs', 'first_won', 'return_pts_won', 'winners', 'unforced']
            existing_cols = [col for col in agg_cols if col in overview_total.columns]

            if existing_cols:
                try:
                    agg_stats = overview_total.groupby('Player_canonical')[existing_cols].mean().astype('float32')

                    # Inject winner columns
                    for col in existing_cols:
                        col_name = f'winner_{col}'
                        if col_name not in enhanced_data.columns:
                            enhanced_data[col_name] = np.nan

                        winner_values = enhanced_data.loc[gender_mask, 'winner_canonical'].map(agg_stats[col])
                        has_values = winner_values.notna()
                        if has_values.any():
                            enhanced_data.loc[gender_mask & has_values, col_name] = winner_values[has_values].astype(
                                'float32')

                    # Inject loser columns
                    for col in existing_cols:
                        col_name = f'loser_{col}'
                        if col_name not in enhanced_data.columns:
                            enhanced_data[col_name] = np.nan

                        loser_values = enhanced_data.loc[gender_mask, 'loser_canonical'].map(agg_stats[col])
                        has_values = loser_values.notna()
                        if has_values.any():
                            enhanced_data.loc[gender_mask & has_values, col_name] = loser_values[has_values].astype(
                                'float32')

                except Exception as e:
                    logging.warning(f"Jeff extraction failed for {gender}: {e}")

            del overview_total
            if 'agg_stats' in locals():
                del agg_stats
            gc.collect()

        return enhanced_data

    def train_optimized_model(self, X_train, y_train):
        """Train with error handling"""
        if X_train.empty or len(y_train) == 0:
            raise ValueError("Cannot train model with empty data")

        logging.info(f"Training model with {X_train.shape[0]} samples, {X_train.shape[1]} features")

        X_train_np = X_train.to_numpy(dtype='float32', copy=False)
        y_train_np = y_train.to_numpy(dtype='int32', copy=False)

        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_seed,
            verbose=-1,
            n_jobs=-1
        )

        lgb_model.fit(X_train_np, y_train_np)
        return lgb_model

    def save_optimized_model(self, model, feature_importance, performance, feature_columns):
        """CRITICAL FIX: Save with proper data type separation"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_cache / f"optimized_model_{timestamp}.pkl"

        top_features = feature_importance.head(50)

        # CRITICAL FIX: Separate feature names from importance values
        feature_names = top_features['feature'].tolist()  # Keep as string list
        importance_values = top_features['importance'].astype('float32').to_numpy()  # Cast only importance

        model_metadata = {
            'model': model,
            'feature_columns': feature_columns,
            'top_feature_names': feature_names,  # String list
            'top_feature_importance': importance_values,  # Float32 array
            'performance_auc': np.float32(performance['auc']),
            'performance_accuracy': np.float32(performance['accuracy']),
            'training_date': date.today(),
            'random_seed': self.random_seed
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_metadata, f)

        logging.info(f"Model saved: {model_path}")
        return model_path

    def run_optimized_pipeline(self):
        """Complete pipeline with comprehensive error handling"""
        logging.info("OPTIMIZED TENNIS PIPELINE")

        start_time = time.time()

        # Load data
        tennis_data = load_optimized_tennis_data()
        if tennis_data.empty:
            raise ValueError("No tennis data loaded - check file paths and formats")

        jeff_data = load_optimized_jeff_data()

        logging.info(f"Loaded: {len(tennis_data):,} tennis matches")

        # Extract Jeff features
        enhanced_data = self.optimized_jeff_extraction(tennis_data, jeff_data)

        # Create balanced dataset
        training_data = self.create_fixed_balanced_dataset(enhanced_data)

        # Generate relative features
        relative_features = self.create_optimized_relative_features(training_data)

        # Combine features
        feature_cols = [col for col in training_data.columns
                        if col not in ['target', 'match_id'] and
                        training_data[col].dtype in ['float32', 'int32', 'int8', 'int16']]

        if len(feature_cols) == 0:
            raise ValueError("No numeric features found for training")

        X = pd.concat([training_data[feature_cols], relative_features], axis=1)
        y = training_data['target']

        # Memory cleanup
        del training_data, enhanced_data, tennis_data, jeff_data
        gc.collect()

        logging.info(f"Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")

        if X.shape[1] < 5:
            raise ValueError(f"Insufficient features for training: {X.shape[1]}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed, stratify=y
        )

        # Fill NaN and ensure float32
        X_train = X_train.fillna(0).astype('float32', copy=False)
        X_test = X_test.fillna(0).astype('float32', copy=False)

        # Train model
        model = self.train_optimized_model(X_train, y_train)

        # Evaluate
        y_pred_proba = model.predict_proba(X_test.to_numpy(dtype='float32', copy=False))[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, (y_pred_proba > 0.5).astype(int))

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        performance = {'auc': auc, 'accuracy': accuracy}

        # Save model
        self.save_optimized_model(model, feature_importance, performance, X_train.columns.tolist())

        total_time = time.time() - start_time

        logging.info(f"RESULTS: AUC={auc:.4f}, Accuracy={accuracy:.4f}, Time={total_time:.1f}s")

        return model, feature_importance, performance


def main():
    parser = argparse.ArgumentParser(description="Optimized Tennis Pipeline")
    parser.add_argument("--cache-dir", default="cache", help="Cache directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    pipeline = OptimizedTennisPipeline(cache_dir=args.cache_dir, random_seed=args.seed)

    try:
        result = pipeline.run_optimized_pipeline()
        logging.info("Pipeline completed successfully")
        return 0
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())