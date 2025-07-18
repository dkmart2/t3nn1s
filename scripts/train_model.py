import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from tennis_updated import AutomatedTennisAbstractScraper, integrate_scraped_data_hybrid
from tennis_updated import generate_comprehensive_historical_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from datetime import datetime
from tennis_updated import extract_comprehensive_jeff_features

def main():
    # Scrape matches for API-enhanced data from cutoff onward
    scraper = AutomatedTennisAbstractScraper()
    records = scraper.automated_scraping_session(force=False)

    # Generate full Jeff history
    hist_full, jeff_data_full, defaults_full = generate_comprehensive_historical_data(fast=False)

    # Split Jeff data before cutoff date (2025-06-10)
    cutoff = datetime(2025, 6, 10)
    jeff_rows = []
    for _, row in hist_full.iterrows():
        match_date = datetime.strptime(row['composite_id'][:8], '%Y%m%d')
        if match_date < cutoff:
            w_feats = extract_comprehensive_jeff_features(
                row['winner_canonical'], row.get('gender', 'M'), jeff_data_full.get(row['winner_canonical'], {})
            )
            l_feats = extract_comprehensive_jeff_features(
                row['loser_canonical'], row.get('gender', 'M'), jeff_data_full.get(row['loser_canonical'], {})
            )
            combined = {
                'composite_id': row['composite_id'],
                'winner_canonical': row['winner_canonical'],
                'loser_canonical': row['loser_canonical'],
                **{'winner_' + k: v for k, v in w_feats.items()},
                **{'loser_' + k: v for k, v in l_feats.items()}
            }
            jeff_rows.append(combined)
    df_jeff = pd.DataFrame(jeff_rows)

    # Integrate API-Tennis enhanced data for matches on/after cutoff
    hist_api = hist_full[hist_full['composite_id'].str[:8] >= cutoff.strftime('%Y%m%d')]
    df_api = integrate_scraped_data_hybrid(hist_api, records)

    # Combine datasets
    df = pd.concat([df_jeff, df_api], ignore_index=True)

    # Determine TA feature columns; fallback to all numeric Jeff features if none
    feat_cols = [c for c in df.columns if c.startswith(("winner_ta_","loser_ta_"))]
    if not feat_cols:
        # Use all numeric features except label for Jeff-based training
        feat_cols = [
            c for c in df.select_dtypes(include=['number']).columns
            if c not in ['label']
        ]
    X = df[feat_cols].fillna(0)
    # Labels: winner column always 1
    df['label'] = 1  # since df rows correspond to actual winners from historical
    y = df['label']

    # Handle case of too few samples for train/test split
    n_samples = len(X)
    if n_samples < 2:
        print(f"Not enough samples ({n_samples}) to train model. Exiting.")
        return

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluation
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Log Loss:", log_loss(y_test, probs))

if __name__ == "__main__":
    main()
