import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path


def prepare_point_data_from_tennis_abstract(scraped_records: list) -> pd.DataFrame:
    """Convert Tennis Abstract scraped records to point-level training data"""

    # Filter for point-level records
    point_records = [r for r in scraped_records if 'match_id' in r and 'Pt' in r]

    if not point_records:
        return pd.DataFrame()

    point_df = pd.DataFrame(point_records)

    # Merge with match metadata
    match_records = [r for r in scraped_records if r.get('data_type') == 'match_metadata']
    match_df = pd.DataFrame(match_records)

    # Join on match_id
    point_df = point_df.merge(match_df[['match_id', 'surface', 'tournament', 'date']],
                              on='match_id', how='left')

    # Add player strength features (would come from Elo or rankings)
    # This is placeholder - integrate with your jeff_data
    point_df['server_elo'] = 1500
    point_df['returner_elo'] = 1500
    point_df['server_h2h_win_pct'] = 0.5

    return point_df


def prepare_match_features_for_ensemble(historical_data: pd.DataFrame) -> pd.DataFrame:
    """Prepare match-level features for ensemble training"""

    features = historical_data.copy()

    # Add derived features
    features['winner_serve_pct'] = (
                                           features['winner_first_won'] + features['winner_second_won']
                                   ) / features['winner_serve_pts'].clip(lower=1)

    features['loser_serve_pct'] = (
                                          features['loser_first_won'] + features['loser_second_won']
                                  ) / features['loser_serve_pts'].clip(lower=1)

    # Add actual winner column (always 1 since winner is P1 by definition)
    features['actual_winner'] = 1

    # Add placeholder columns for demonstration
    features['winner_last10_wins'] = 6  # Would calculate from recent matches
    features['loser_last10_wins'] = 4
    features['p1_surface_h2h_wins'] = features['p1_h2h_wins']
    features['p2_surface_h2h_wins'] = features['p2_h2h_wins']

    return features


def retrain_models_daily(tennis_updated_module):
    """Daily retraining routine"""

    logging.info("Starting daily model retraining...")

    # Load latest data
    hist, jeff_data, defaults = tennis_updated_module.load_from_cache_with_scraping()

    # Get point-level data from Tennis Abstract
    scraper = tennis_updated_module.AutomatedTennisAbstractScraper()
    point_records = scraper.automated_scraping_session(days_back=30, max_matches=1000)
    point_df = prepare_point_data_from_tennis_abstract(point_records)

    # Get match-level features
    match_df = prepare_match_features_for_ensemble(hist)

    # Initialize and train pipeline
    pipeline = TennisModelPipeline()

    if not point_df.empty:
        pipeline.train(point_df, match_df)

        # Save model
        model_path = Path("models") / f"tennis_model_{datetime.now().strftime('%Y%m%d')}.pkl"
        model_path.parent.mkdir(exist_ok=True)
        pipeline.save(str(model_path))

        logging.info(f"Model saved to {model_path}")
    else:
        logging.warning("No point data available for training")

    return pipeline


def predict_match_v2(args, hist, jeff_data, defaults):
    """Enhanced prediction function using the new model"""

    # Load latest trained model
    model_path = sorted(Path("models").glob("tennis_model_*.pkl"))[-1]
    pipeline = TennisModelPipeline()
    pipeline.load(str(model_path))

    # Find match in historical data (reuse existing logic)
    match_date = pd.to_datetime(args.date).date()
    comp_id = f"{match_date.strftime('%Y%m%d')}-{args.tournament.lower()}-{args.player1.lower()}-{args.player2.lower()}"

    row = hist[hist["composite_id"] == comp_id]

    if row.empty:
        print("Match not found in historical data")
        return None

    match_data = row.iloc[0].to_dict()

    # Build match context for new model
    match_context = {
        'surface': match_data.get('surface', 'Hard'),
        'WRank': match_data.get('WRank', 50),
        'LRank': match_data.get('LRank', 50),
        'winner_elo': match_data.get('winner_jeff_elo', 1500),
        'loser_elo': match_data.get('loser_jeff_elo', 1500),
        'elo_diff': match_data.get('winner_jeff_elo', 1500) - match_data.get('loser_jeff_elo', 1500),
        'h2h_advantage': match_data.get('p1_h2h_win_pct', 0.5) - 0.5,
        'p1_h2h_win_pct': match_data.get('p1_h2h_win_pct', 0.5),
        'winner_serve_pts': match_data.get('winner_serve_pts', 80),
        'loser_serve_pts': match_data.get('loser_serve_pts', 80),
        'winner_aces': match_data.get('winner_aces', 6),
        'loser_aces': match_data.get('loser_aces', 6),
        'winner_last10_wins': 6,  # Placeholder
        'loser_last10_wins': 4,  # Placeholder
        'p1_surface_h2h_wins': match_data.get('p1_h2h_wins', 0),
        'p2_surface_h2h_wins': match_data.get('p2_h2h_wins', 0),
        'tournament_tier': match_data.get('tournament_tier', 'ATP250'),
        'data_quality_score': match_data.get('data_quality_score', 0.5)
    }

    # Make prediction
    result = pipeline.predict(match_context)

    print(f"\n=== ENHANCED PREDICTION RESULTS ===")
    print(f"Win Probability: {result['win_probability']:.3f}")
    print(f"  - Simulation Component: {result['simulation_component']:.3f}")
    print(f"  - Direct ML Component: {result['direct_component']:.3f}")
    print(f"Confidence: {result['confidence']}")

    return result['win_probability']


# Production automation script
def production_pipeline():
    """Automated daily pipeline"""

    import tennis_updated
    from datetime import date

    logging.basicConfig(
        filename=f'logs/model_training_{date.today()}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # 1. Update data
        logging.info("Updating historical data...")
        hist, jeff_data, defaults = tennis_updated.load_from_cache_with_scraping()

        if hist is None:
            hist, jeff_data, defaults = tennis_updated.generate_comprehensive_historical_data(fast=False)
            hist = tennis_updated.run_automated_tennis_abstract_integration(hist)
            tennis_updated.save_to_cache(hist, jeff_data, defaults)

        # 2. Integrate latest API data
        logging.info("Integrating API data...")
        hist = tennis_updated.integrate_api_tennis_data_incremental(hist)
        tennis_updated.save_to_cache(hist, jeff_data, defaults)

        # 3. Retrain models
        logging.info("Retraining models...")
        pipeline = retrain_models_daily(tennis_updated)

        # 4. Generate predictions for today's matches
        logging.info("Generating predictions for today...")
        today_fixtures = tennis_updated.get_fixtures_for_date(date.today())

        predictions = []
        for fixture in today_fixtures:
            if fixture.get('event_status') == 'Not Started':
                try:
                    args = type('Args', (), {
                        'player1': fixture.get('event_first_player'),
                        'player2': fixture.get('event_second_player'),
                        'date': date.today().isoformat(),
                        'tournament': fixture.get('tournament_name'),
                        'best_of': 3
                    })

                    prob = predict_match_v2(args, hist, jeff_data, defaults)

                    if prob:
                        predictions.append({
                            'match': f"{args.player1} vs {args.player2}",
                            'probability': prob,
                            'tournament': args.tournament
                        })
                except Exception as e:
                    logging.error(f"Prediction failed for {fixture}: {e}")

        # 5. Save predictions
        if predictions:
            pred_df = pd.DataFrame(predictions)
            pred_df.to_csv(f'predictions/daily_{date.today()}.csv', index=False)
            logging.info(f"Saved {len(predictions)} predictions")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise


# Evaluation utilities
def evaluate_model_performance(predictions_df: pd.DataFrame) -> dict:
    """Calculate model performance metrics"""

    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt

    metrics = {}

    # Brier score
    y_true = predictions_df['actual_winner'].values
    y_pred = predictions_df['predicted_prob'].values
    metrics['brier_score'] = brier_score_loss(y_true, y_pred)

    # Log loss
    metrics['log_loss'] = log_loss(y_true, y_pred)

    # Calibration
    fraction_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=10)

    # Plot calibration
    plt.figure(figsize=(8, 6))
    plt.plot(mean_pred, fraction_pos, 's-', label='Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('calibration_plot.png')
    plt.close()

    # ROC AUC
    from sklearn.metrics import roc_auc_score
    metrics['roc_auc'] = roc_auc_score(y_true, y_pred)

    # By confidence level
    for conf in ['HIGH', 'MEDIUM', 'LOW']:
        mask = predictions_df['confidence'] == conf
        if mask.sum() > 0:
            metrics[f'brier_{conf.lower()}'] = brier_score_loss(
                y_true[mask], y_pred[mask]
            )

    return metrics