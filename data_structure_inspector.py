#!/usr/bin/env python3
"""
Inspect the actual structure of Jeff Sackmann files
"""


def inspect_jeff_data_structure():
    """Check the actual column structure of Jeff files"""

    print("=== INSPECTING ACTUAL JEFF DATA STRUCTURE ===\n")

    from tennis_updated import load_jeff_comprehensive_data

    jeff_data = load_jeff_comprehensive_data()

    # Check one file in detail
    if 'men' in jeff_data and 'serve_basics' in jeff_data['men']:
        sb_df = jeff_data['men']['serve_basics']

        print("=== SERVE_BASICS FILE STRUCTURE ===")
        print(f"Shape: {sb_df.shape}")
        print(f"Columns: {list(sb_df.columns)}")
        print(f"\nFirst 3 rows:")
        print(sb_df.head(3).to_string())

        # Check unique values in key columns
        if 'row' in sb_df.columns:
            print(f"\nUnique 'row' values: {sb_df['row'].unique()[:10]}")

        if 'match_id' in sb_df.columns:
            print(f"\nSample match_ids: {sb_df['match_id'].unique()[:5]}")

            # Look for Djokovic matches
            djokovic_matches = sb_df[sb_df['match_id'].str.contains('djokovic', case=False, na=False)]
            if not djokovic_matches.empty:
                print(f"\nFound Djokovic matches: {len(djokovic_matches)}")
                print("Sample Djokovic record:")
                print(djokovic_matches.iloc[0].to_dict())

    # Check overview file structure
    if 'men' in jeff_data and 'overview' in jeff_data['men']:
        ov_df = jeff_data['men']['overview']

        print(f"\n=== OVERVIEW FILE STRUCTURE ===")
        print(f"Shape: {ov_df.shape}")
        print(f"Columns: {list(ov_df.columns)}")
        print(f"\nFirst 3 rows:")
        print(ov_df.head(3).to_string())

        # Check if this has Player_canonical
        if 'Player_canonical' in ov_df.columns:
            print(f"\n✅ Overview has 'Player_canonical' column")
            unique_players = ov_df['Player_canonical'].unique()[:10]
            print(f"Sample players: {unique_players}")

            # Look for Djokovic
            djokovic_data = ov_df[ov_df['Player_canonical'] == 'djokovic_n']
            if not djokovic_data.empty:
                print(f"\n✅ Found Djokovic in overview: {len(djokovic_data)} records")
            else:
                print(f"\n❌ No Djokovic found with 'djokovic_n' in overview")
        else:
            print(f"\n❌ Overview does NOT have 'Player_canonical' column")

    # Check a few more files
    other_files = ['net_points', 'rally', 'key_points_serve']

    for file_name in other_files:
        if 'men' in jeff_data and file_name in jeff_data['men']:
            df = jeff_data['men'][file_name]
            print(f"\n=== {file_name.upper()} FILE ===")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")

            # Check if has Player_canonical
            if 'Player_canonical' in df.columns:
                print(f"✅ Has 'Player_canonical'")
            else:
                print(f"❌ No 'Player_canonical' - likely uses match_id structure")


def check_match_id_structure():
    """Understand how match_id maps to players"""

    print(f"\n=== UNDERSTANDING MATCH_ID STRUCTURE ===\n")

    from tennis_updated import load_jeff_comprehensive_data

    jeff_data = load_jeff_comprehensive_data()

    # Check matches file to understand the mapping
    if 'men' in jeff_data and 'matches' in jeff_data['men']:
        matches_df = jeff_data['men']['matches']

        print("=== MATCHES FILE STRUCTURE ===")
        print(f"Shape: {matches_df.shape}")
        print(f"Columns: {list(matches_df.columns)}")
        print(f"\nFirst 3 rows:")
        print(matches_df.head(3).to_string())

        # Look for Djokovic matches
        if 'winner_canonical' in matches_df.columns:
            djokovic_matches = matches_df[
                (matches_df['winner_canonical'] == 'djokovic_n') |
                (matches_df['loser_canonical'] == 'djokovic_n')
                ]

            if not djokovic_matches.empty:
                print(f"\n✅ Found Djokovic matches: {len(djokovic_matches)}")
                sample_match = djokovic_matches.iloc[0]
                sample_match_id = sample_match['match_id']
                print(f"Sample match_id: {sample_match_id}")

                # Now check if this match_id appears in serve_basics
                if 'serve_basics' in jeff_data['men']:
                    sb_df = jeff_data['men']['serve_basics']
                    match_data = sb_df[sb_df['match_id'] == sample_match_id]

                    if not match_data.empty:
                        print(f"\n✅ Found match in serve_basics: {len(match_data)} rows")
                        print("Sample rows:")
                        print(match_data.to_string())
                    else:
                        print(f"\n❌ Match not found in serve_basics")


if __name__ == "__main__":
    inspect_jeff_data_structure()
    check_match_id_structure()