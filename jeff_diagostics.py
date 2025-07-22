#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Import your functions
sys.path.append('.')
from tennis_updated import load_all_tennis_data, load_jeff_comprehensive_data


# Import the enhanced canonicalization function from your fixed pipeline
def enhanced_canon_name(name: str) -> str:
    """Same function as in the main pipeline"""
    if pd.isna(name) or name == '':
        return ""

    from unidecode import unidecode
    import re

    s = unidecode(str(name)).lower()
    s = s.replace('-', ' ')
    s = re.sub(r'[^a-z0-9\s]+', '', s)
    tokens = [t for t in s.split() if t]

    if not tokens:
        return ""

    if len(tokens) == 1:
        return tokens[0]

    elif len(tokens) == 2:
        first, second = tokens

        has_dot_in_original = '.' in name
        is_very_short_second = len(second) <= 2
        looks_like_initials = len(second) <= 4 and has_dot_in_original

        if is_very_short_second or looks_like_initials:
            return f"{first}_{second}"
        else:
            return f"{second}_{first[0]}"

    else:
        if '.' in name:
            surname_tokens = []
            initial_tokens = []

            for token in tokens:
                if len(token) <= 2:
                    initial_tokens.append(token)
                else:
                    surname_tokens.append(token)

            if surname_tokens and initial_tokens:
                surname = '_'.join(surname_tokens)
                initials = ''.join(initial_tokens)
                return f"{surname}_{initials}"

        surname = tokens[-1]
        given_names = tokens[:-1]
        initials = ''.join(name[0] for name in given_names if name)

        return f"{surname}_{initials}"


def diagnose_jeff_feature_extraction():
    """Comprehensive diagnosis of Jeff feature extraction"""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    print("🔍 JEFF FEATURE EXTRACTION DIAGNOSTIC")
    print("=" * 50)

    # 1. Load data
    print("\n1. Loading data...")
    tennis_data = load_all_tennis_data()
    jeff_data = load_jeff_comprehensive_data()

    print(f"   Tennis data: {len(tennis_data):,} matches")
    print(f"   Jeff data loaded: {bool(jeff_data)}")

    # 2. Check tennis data structure
    print("\n2. Tennis data structure:")
    print(f"   Columns: {tennis_data.columns.tolist()}")
    print(f"   Sample winners: {tennis_data['Winner'].head(3).tolist()}")
    print(f"   Sample losers: {tennis_data['Loser'].head(3).tolist()}")

    # Add canonicalization
    tennis_data['winner_canonical'] = tennis_data['Winner'].apply(enhanced_canon_name)
    tennis_data['loser_canonical'] = tennis_data['Loser'].apply(enhanced_canon_name)

    print(f"   Sample canonical winners: {tennis_data['winner_canonical'].head(3).tolist()}")
    print(f"   Sample canonical losers: {tennis_data['loser_canonical'].head(3).tolist()}")

    # 3. Check Jeff data structure
    print("\n3. Jeff data structure:")
    for gender in ['men', 'women']:
        if gender in jeff_data:
            print(f"   {gender.upper()}:")
            for dataset_name, df in jeff_data[gender].items():
                if df is not None and not df.empty:
                    print(f"     {dataset_name}: {len(df)} rows, {len(df.columns)} columns")
                    if 'player' in df.columns:
                        sample_players = df['player'].head(3).tolist()
                        sample_canonical = df['player'].apply(enhanced_canon_name).head(3).tolist()
                        print(f"       Sample players: {sample_players}")
                        print(f"       Sample canonical: {sample_canonical}")

    # 4. Check overlap analysis
    print("\n4. Detailed overlap analysis:")
    tennis_canonical = set(tennis_data['winner_canonical']) | set(tennis_data['loser_canonical'])
    tennis_canonical.discard('')  # Remove empty strings

    print(f"   Tennis canonical players: {len(tennis_canonical)}")

    for gender in ['men', 'women']:
        if gender in jeff_data and 'overview' in jeff_data[gender]:
            overview_df = jeff_data[gender]['overview']
            if 'player' in overview_df.columns:
                jeff_canonical = set(overview_df['player'].apply(enhanced_canon_name))
                jeff_canonical.discard('')

                overlap = tennis_canonical & jeff_canonical
                print(f"   {gender} Jeff canonical players: {len(jeff_canonical)}")
                print(
                    f"   {gender} overlap: {len(overlap)} players ({len(overlap) / len(tennis_canonical) * 100:.1f}%)")

                # Show specific overlapping players
                sample_overlap = list(overlap)[:10]
                print(f"   {gender} overlapping players: {sample_overlap}")

    # 5. Check Jeff aggregation process
    print("\n5. Jeff aggregation diagnostic:")

    # Simple aggregation test
    for gender in ['men', 'women']:
        if gender in jeff_data and 'overview' in jeff_data[gender]:
            overview_df = jeff_data[gender]['overview']
            print(f"   {gender} overview before processing:")
            print(f"     Shape: {overview_df.shape}")
            print(f"     Columns: {overview_df.columns.tolist()}")

            if 'player' in overview_df.columns:
                overview_df = overview_df.copy()
                overview_df['Player_canonical'] = overview_df['player'].apply(enhanced_canon_name)

                # Check for 'Total' filter
                if 'set' in overview_df.columns:
                    set_values = overview_df['set'].value_counts()
                    print(f"     'set' column values: {set_values.to_dict()}")

                    overview_total = overview_df[overview_df['set'] == 'Total']
                    print(f"     After 'Total' filter: {len(overview_total)} rows")
                else:
                    overview_total = overview_df
                    print(f"     No 'set' column, using all rows")

                # Check aggregation columns
                agg_cols = ['serve_pts', 'aces', 'dfs', 'first_in', 'first_won',
                            'second_won', 'bp_saved', 'return_pts_won', 'winners',
                            'winners_fh', 'winners_bh', 'unforced', 'unforced_fh', 'unforced_bh']
                existing_cols = [col for col in agg_cols if col in overview_total.columns]
                print(f"     Available agg columns: {existing_cols}")

                if existing_cols and not overview_total.empty:
                    # Perform aggregation
                    result = overview_total.groupby('Player_canonical')[existing_cols].mean()
                    print(f"     Aggregation result shape: {result.shape}")
                    print(f"     Sample aggregated data:")
                    print(result.head(3))

                    # Check for NaN values
                    nan_counts = result.isnull().sum()
                    print(f"     NaN counts per column: {nan_counts.to_dict()}")

    # 6. Test a manual merge
    print("\n6. Manual merge test:")

    # Take a small subset of tennis data for testing
    test_tennis = tennis_data.head(100).copy()

    # Add gender (simple version)
    test_tennis['gender'] = 'M'  # Assume men for testing

    print(f"   Test tennis subset: {len(test_tennis)} matches")
    print(f"   Sample canonical winners: {test_tennis['winner_canonical'].head(5).tolist()}")

    # Try to merge with Jeff data
    if 'men' in jeff_data and 'overview' in jeff_data['men']:
        overview_df = jeff_data['men']['overview']
        if 'player' in overview_df.columns:
            overview_df = overview_df.copy()
            overview_df['Player_canonical'] = overview_df['player'].apply(enhanced_canon_name)

            if 'set' in overview_df.columns:
                overview_total = overview_df[overview_df['set'] == 'Total']
            else:
                overview_total = overview_df

            if not overview_total.empty:
                agg_cols = ['serve_pts', 'aces', 'dfs', 'first_in', 'first_won']
                existing_cols = [col for col in agg_cols if col in overview_total.columns]

                if existing_cols:
                    result = overview_total.groupby('Player_canonical')[existing_cols].mean()

                    # Try merge
                    prefixed_result = result.add_prefix('winner_')
                    merge_test = test_tennis.merge(
                        prefixed_result,
                        left_on='winner_canonical',
                        right_index=True,
                        how='left'
                    )

                    print(f"   Merge test result shape: {merge_test.shape}")
                    print(f"   Merge test columns: {merge_test.columns.tolist()}")

                    # Check for actual matches
                    if len(prefixed_result.columns) > 0:
                        first_col = prefixed_result.columns[0]
                        matches = merge_test[first_col].notna().sum()
                        print(
                            f"   Successful merges: {matches}/{len(test_tennis)} ({matches / len(test_tennis) * 100:.1f}%)")

                        if matches > 0:
                            # Show successful matches
                            successful_rows = merge_test[merge_test[first_col].notna()]
                            print(f"   Sample successful matches:")
                            for _, row in successful_rows.head(3).iterrows():
                                print(f"     {row['Winner']} → {row['winner_canonical']} → {row[first_col]:.3f}")

    print("\n" + "=" * 50)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    diagnose_jeff_feature_extraction()