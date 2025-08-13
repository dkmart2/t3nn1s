import pandas as pd
import os
from settings import TENNIS_CACHE_DIR

# Load cache
hd_path = os.path.join(TENNIS_CACHE_DIR, 'historical_data.parquet')
df = pd.read_parquet(hd_path)
df['date'] = pd.to_datetime(df['date'])

print('=== DATA CLEANUP ===')

# Check for matches with real statistics
has_aces = pd.notna(df['winner_aces']) & (df['winner_aces'] > 0)
has_points = pd.notna(df['winner_total_points']) & (df['winner_total_points'] > 0) if 'winner_total_points' in df.columns else pd.Series([False]*len(df))
has_real_stats = has_aces | has_points

print(f'Total matches: {len(df)}')
print(f'Matches with real stats: {has_real_stats.sum()}')
print(f'Matches without real stats: {(~has_real_stats).sum()}')

# Get last real match
if has_real_stats.sum() > 0:
    real_matches = df[has_real_stats].sort_values('date')
    last_real = real_matches.iloc[-1]
    print(f'\n✅ Last match with REAL stats:')
    print(f'   Date: {last_real["date"].date()}')
    print(f'   Match: {last_real.get("Winner", "?")} def. {last_real.get("Loser", "?")}')
    
    # Check for future placeholders
    today = pd.Timestamp.now().normalize()
    future_no_stats = (df['date'] >= today) & (~has_real_stats)
    
    print(f'\n⚠️  Found {future_no_stats.sum()} future/today placeholder matches')
    
    if future_no_stats.sum() > 0:
        # Show what we're removing
        placeholders = df[future_no_stats]
        print('\nPlaceholder matches to remove:')
        for idx, row in placeholders.iterrows():
            print(f'   {row["date"].date()}: {row.get("Winner", "?")} vs {row.get("Loser", "?")}')
        
        # Remove them
        df_clean = df[~future_no_stats]
        df_clean.to_parquet(hd_path, index=False)
        print(f'\n✅ Dataset cleaned!')
        print(f'   Before: {len(df)} matches')
        print(f'   After: {len(df_clean)} matches')
        print(f'   Latest date: {df_clean["date"].max().date()}')
    else:
        print('✅ No placeholder matches to remove')
else:
    print('\n❌ WARNING: No matches with real statistics found!')