import pandas as pd
from data_integration import integrate_scraped_data_hybrid
from model import BayesianTennisModel

def train_model(historical, records):
    df = integrate_scraped_data_hybrid(historical, records)
    # Temporary placeholder: set predicted_winner to actual winner to avoid missing column
    df['predicted_winner'] = df['winner_canonical']
    X = df.drop(columns=["winner_canonical", "predicted_winner"])
    y = (df["winner_canonical"] == df["predicted_winner"]).astype(int)
    model = BayesianTennisModel()
    model.fit(X, y)
    return model
2