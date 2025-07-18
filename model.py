import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss
import lightgbm as lgb
from typing import Dict, Tuple, Optional
import joblib
from sklearn.calibration import CalibratedClassifierCV
from tennis_updated import TennisAbstractScraper

#%%
class PointLevelModel:
    """Learns P(point won | features) from historical point data"""

    def __init__(self):
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = None

    def engineer_point_features(self, point_data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from raw point data"""
        features = pd.DataFrame()

        # Serve features
        features['is_first_serve'] = point_data['2nd'].isna().astype(int)
        features['serve_direction_wide'] = point_data['1st'].str.contains('w', na=False).astype(int)
        features['serve_direction_body'] = point_data['1st'].str.contains('b', na=False).astype(int)
        features['serve_direction_t'] = point_data['1st'].str.contains('t', na=False).astype(int)

        # Rally features
        features['rally_length'] = point_data['rallyCount'].fillna(1)
        features['is_net_point'] = (point_data['1st'].str.contains('@', na=False) |
                                    point_data['2nd'].str.contains('@', na=False)).astype(int)

        # Score state
        features['games_diff'] = point_data['Gm1'] - point_data['Gm2']
        features['sets_diff'] = point_data['Set1'] - point_data['Set2']
        features['is_tiebreak'] = point_data['TbSet'].astype(int)

        # Point importance
        point_score = point_data['Pts'].fillna('0-0')
        features['is_break_point'] = ((point_score.isin(['0-40', '15-40', '30-40', 'AD-40'])) &
                                      (point_data['Svr'] == 2)).astype(int)
        features['is_game_point'] = point_score.isin(['40-0', '40-15', '40-30', '40-AD']).astype(int)

        # Surface (from match metadata)
        features['surface_clay'] = (point_data['surface'] == 'Clay').astype(int)
        features['surface_grass'] = (point_data['surface'] == 'Grass').astype(int)
        features['surface_hard'] = (point_data['surface'] == 'Hard').astype(int)

        # Player strength differential (from match metadata)
        features['elo_diff'] = point_data['server_elo'] - point_data['returner_elo']
        features['h2h_server_advantage'] = point_data['server_h2h_win_pct'] - 0.5

        self.feature_names = features.columns.tolist()
        return features

    def fit(self, point_data: pd.DataFrame):
        """Train the point-level model"""
        X = self.engineer_point_features(point_data)
        y = (point_data['PtWinner'] == point_data['Svr']).astype(int)  # Server wins point

        # Remove NaN rows
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.model.fit(X_scaled, y)

        # Calibrate point-win probabilities with isotonic regression
        calib = CalibratedClassifierCV(self.model, method='isotonic', cv='prefit')
        calib.fit(X_scaled, y)
        self.model = calib

        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance

    def predict_proba(self, point_features: pd.DataFrame) -> np.ndarray:
        """Predict point-win probability"""
        X_scaled = self.scaler.transform(point_features)
        return self.model.predict_proba(X_scaled)[:, 1]


class StateDependentModifiers:
    """Momentum and pressure adjustments"""

    def __init__(self):
        self.momentum_decay = 0.85
        self.pressure_multipliers = {}

    def calculate_momentum(self, recent_points: list, player: int) -> float:
        """Calculate momentum based on recent point outcomes"""
        if not recent_points:
            return 0.0

        weights = np.array([self.momentum_decay ** i for i in range(len(recent_points))])
        player_wins = np.array([1 if p == player else -1 for p in recent_points])

        momentum = np.sum(weights * player_wins) / np.sum(weights)
        return np.tanh(momentum * 0.3)  # Bounded [-1, 1]

    def get_pressure_modifier(self, score_state: dict) -> float:
        """Get pressure modifier based on score state"""
        if score_state.get('is_match_point'):
            return self.pressure_multipliers.get('match_point', 1.0)
        elif score_state.get('is_set_point'):
            return self.pressure_multipliers.get('set_point', 1.0)
        elif score_state.get('is_break_point'):
            return self.pressure_multipliers.get('break_point', 1.0)
        return 1.0

    def fit(self, point_data: pd.DataFrame):
        """
        Learn pressure multipliers from historical point data.
        Expects columns: 'is_break_point', 'is_set_point', 'is_match_point',
        'PtWinner', 'Svr' (server id 1/2).
        """
        # Baseline server win probability
        overall = (point_data['PtWinner'] == point_data['Svr']).mean()
        # Compute conditional probabilities
        for label in ['break_point', 'set_point', 'match_point']:
            mask = point_data[f'is_{label}'] == 1
            if mask.any():
                p = (point_data[mask]['PtWinner'] == point_data[mask]['Svr']).mean()
                # Multiplier = p / overall
                self.pressure_multipliers[label] = p / overall if overall > 0 else 1.0
            else:
                self.pressure_multipliers[label] = 1.0

    def fit_momentum(self, point_data: pd.DataFrame):
        """
        Learn momentum decay parameter from historical point-by-point data.
        Dynamically detects server and winner columns.
        """
        cols = list(point_data.columns)
        # Skip if raw pointlog columns are absent
        if not any(c.lower() == 'svr' for c in cols) or not any('winner' in c.lower() and c.lower() != 'svr' for c in cols):
            print("fit_momentum: missing raw Svr/PtWinner columns, skipping momentum learning")
            return

        # Find server and winner columns
        server_col = next((c for c in cols if c.lower() == 'svr'), None)
        winner_col = next((c for c in cols if 'winner' in c.lower() and c.lower() != 'svr'), None)

        outcomes = (point_data[winner_col] == point_data[server_col]).astype(int).tolist()
        best_decay = self.momentum_decay
        best_corr = -float('inf')
        # Search decay values between 0.5 and 0.99
        for d in np.linspace(0.5, 0.99, 10):
            momentums = []
            for j in range(len(point_data)):
                server = point_data[server_col].iloc[j]
                prev_winners = point_data[winner_col].iloc[:j].tolist()
                # Compute weighted sign series
                if j > 0:
                    weights = np.array([d ** (j - k - 1) for k in range(j)])
                    signs = np.array([1 if w == server else -1 for w in prev_winners])
                    m = (weights * signs).sum() / weights.sum()
                else:
                    m = 0.0
                momentums.append(m)
            # Compute Pearson correlation between momentum and actual outcomes
            corr = np.corrcoef(momentums, outcomes)[0, 1] if len(momentums) > 1 else 0
            if corr > best_corr:
                best_corr = corr
                best_decay = d
        self.momentum_decay = best_decay
        print(f"Learned momentum_decay = {self.momentum_decay:.3f} (corr={best_corr:.3f})")


class DataDrivenTennisModel:
    """Enhanced model with ML-based point probabilities"""

    def __init__(self, point_model: PointLevelModel, n_simulations: int = 1000):
        self.point_model = point_model
        self.n_simulations = n_simulations
        self.state_modifiers = StateDependentModifiers()
        self.recent_points = []  # Track for momentum

    def get_point_win_prob(self, match_context: dict, score_state: dict,
                           momentum: dict) -> float:
        """Get point-win probability from trained model"""
        # Build feature vector for current point
        features = pd.DataFrame([{
            'is_first_serve': 1,  # Simplification - would track actual serve
            'serve_direction_wide': 0.3,  # Would come from player tendencies
            'serve_direction_body': 0.3,
            'serve_direction_t': 0.4,
            'rally_length': 4.5,  # Expected rally length
            'is_net_point': 0,
            'games_diff': score_state['games_diff'],
            'sets_diff': score_state['sets_diff'],
            'is_tiebreak': score_state['is_tiebreak'],
            'is_break_point': score_state['is_break_point'],
            'is_game_point': score_state['is_game_point'],
            'surface_clay': match_context['surface'] == 'Clay',
            'surface_grass': match_context['surface'] == 'Grass',
            'surface_hard': match_context['surface'] == 'Hard',
            'elo_diff': match_context['elo_diff'],
            'h2h_server_advantage': match_context['h2h_advantage']
        }])

        # Get base probability from model
        base_prob = self.point_model.predict_proba(features)[0]

        # Apply state-dependent modifiers
        pressure_mod = self.state_modifiers.get_pressure_modifier(score_state)
        momentum_mod = 1 + momentum['server'] * 0.05  # ±5% for momentum

        # Combine modifiers
        adjusted_prob = base_prob * pressure_mod * momentum_mod

        # Ensure valid probability
        return np.clip(adjusted_prob, 0.01, 0.99)

    def simulate_game(self, match_context: dict, score_state: dict, server: int) -> int:
        """Simulate game with dynamic point probabilities"""
        points = {'server': 0, 'returner': 0}
        momentum = {'server': 0, 'returner': 0}

        while True:
            # Calculate current momentum
            momentum['server'] = self.state_modifiers.calculate_momentum(
                self.recent_points[-10:], server
            )
            momentum['returner'] = -momentum['server']

            # Update score state
            score_state['is_break_point'] = (
                    points['returner'] >= 3 and
                    points['returner'] > points['server'] and
                    points['returner'] - points['server'] >= 1
            )
            score_state['is_game_point'] = (
                    points['server'] >= 3 and
                    points['server'] > points['returner']
            )

            # Get point probability
            point_prob = self.get_point_win_prob(match_context, score_state, momentum)

            # Simulate point
            if np.random.random() < point_prob:
                points['server'] += 1
                self.recent_points.append(server)
            else:
                points['returner'] += 1
                self.recent_points.append(3 - server)  # Other player

            # Check game end
            if (points['server'] >= 4 or points['returner'] >= 4) and \
                    abs(points['server'] - points['returner']) >= 2:
                return server if points['server'] > points['returner'] else 3 - server

    def simulate_match(self, match_context: dict, best_of: int = 3) -> float:
        """Run Monte Carlo simulation with learned probabilities"""
        wins = 0

        # Determine number of sets required to win
        sets_to_win = best_of // 2 + 1

        for _ in range(self.n_simulations):
            self.recent_points = []  # Reset momentum tracking
            p1_sets = p2_sets = 0

            while p1_sets < sets_to_win and p2_sets < sets_to_win:
                # Simulate set...
                set_winner = self._simulate_set(match_context, p1_sets, p2_sets)

                if set_winner == 1:
                    p1_sets += 1
                else:
                    p2_sets += 1

            if p1_sets > p2_sets:
                wins += 1

        return wins / self.n_simulations

    def _simulate_set(self, match_context: dict, p1_sets: int, p2_sets: int) -> int:
        """Simulate a set with state tracking"""
        p1_games = p2_games = 0
        server = 1

        while True:
            score_state = {
                'games_diff': p1_games - p2_games,
                'sets_diff': p1_sets - p2_sets,
                'is_tiebreak': False,
                'is_break_point': False,
                'is_game_point': False,
                'is_set_point': (p1_games >= 5 or p2_games >= 5) and abs(p1_games - p2_games) >= 1,
                'is_match_point': False  # Would check based on sets and best_of
            }

            game_winner = self.simulate_game(match_context, score_state, server)

            if game_winner == 1:
                p1_games += 1
            else:
                p2_games += 1

            server = 3 - server  # Switch server

            # Set end conditions
            if (p1_games >= 6 or p2_games >= 6) and abs(p1_games - p2_games) >= 2:
                return 1 if p1_games > p2_games else 2
            elif p1_games == 6 and p2_games == 6:
                # Tiebreak - simplified
                return self._simulate_tiebreak(match_context, p1_sets, p2_sets)

    def _simulate_tiebreak(self, match_context: dict, p1_sets: int, p2_sets: int) -> int:
        """Simulate tiebreak"""
        # Simplified - would track actual tiebreak scoring
        tb_prob = self.get_point_win_prob(
            match_context,
            {'is_tiebreak': True, 'sets_diff': p1_sets - p2_sets},
            {'server': 0, 'returner': 0}
        )
        return 1 if np.random.random() < tb_prob else 2


class MatchLevelEnsemble:
    """Direct match prediction + simulation ensemble"""

    def __init__(self):
        self.match_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            colsample_bytree=0.7,
            subsample=0.8
        )
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.ensemble_weights = {'simulation': 0.6, 'direct': 0.4}

    def engineer_match_features(self, match_data: pd.DataFrame) -> pd.DataFrame:
        """Create match-level features"""
        features = pd.DataFrame()

        # Basic features
        features['rank_diff'] = match_data['WRank'] - match_data['LRank']
        features['elo_diff'] = match_data['winner_elo'] - match_data['loser_elo']
        features['h2h_balance'] = match_data['p1_h2h_win_pct'] - 0.5

        # Aggregate stats
        features['winner_serve_dominance'] = (
                match_data['winner_aces'] / match_data['winner_serve_pts'].clip(lower=1)
        )
        features['loser_serve_dominance'] = (
                match_data['loser_aces'] / match_data['loser_serve_pts'].clip(lower=1)
        )

        # Form indicators
        features['winner_recent_win_pct'] = match_data['winner_last10_wins'] / 10
        features['loser_recent_win_pct'] = match_data['loser_last10_wins'] / 10

        # Surface-specific H2H
        features['surface_h2h_diff'] = match_data['p1_surface_h2h_wins'] - match_data['p2_surface_h2h_wins']

        # Tournament importance
        features['is_grand_slam'] = match_data['tournament_tier'].str.contains('Grand Slam', na=False)
        features['is_masters'] = match_data['tournament_tier'].str.contains('Masters', na=False)

        return features

    def fit(self, historical_data: pd.DataFrame):
        """Train ensemble model"""
        X = self.engineer_match_features(historical_data)
        y = (historical_data['actual_winner'] == 1).astype(int)

        # Remove NaN
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]

        # Time-based split for calibration
        split_idx = int(len(X) * 0.8)
        X_train, X_cal = X[:split_idx], X[split_idx:]
        y_train, y_cal = y[:split_idx], y[split_idx:]

        # Train match model
        self.match_model.fit(X_train, y_train)

        # Calibrate
        cal_probs = self.match_model.predict_proba(X_cal)[:, 1]
        self.calibrator.fit(cal_probs, y_cal)

    def predict(self, match_features: pd.DataFrame, simulation_prob: float) -> float:
        """Ensemble prediction"""
        # Direct prediction
        direct_prob = self.match_model.predict_proba(match_features)[:, 1][0]
        direct_prob_cal = self.calibrator.transform([direct_prob])[0]

        # Weighted ensemble
        final_prob = (
                self.ensemble_weights['simulation'] * simulation_prob +
                self.ensemble_weights['direct'] * direct_prob_cal
        )

        return final_prob


class TennisModelPipeline:
    """Complete pipeline orchestrator"""

    def __init__(self):
        self.point_model = PointLevelModel()
        self.match_ensemble = MatchLevelEnsemble()
        self.simulation_model = None

    def train(self, point_data: pd.DataFrame, match_data: pd.DataFrame):
        """Train all components"""
        print("Training point-level model...")
        feature_importance = self.point_model.fit(point_data)
        print(f"Top features:\n{feature_importance.head(10)}")

        print("\nTraining match-level ensemble...")
        self.match_ensemble.fit(match_data)

        print("\nInitializing simulation model...")
        # Fit dynamic pressure multipliers
        print("Learning pressure multipliers from point data...")
        # point_data must include context flags
        self.simulation_model = DataDrivenTennisModel(self.point_model)
        self.simulation_model.state_modifiers.fit(point_data)

        # Instantiate scraper to fetch raw pointlog for momentum learning
        scraper = TennisAbstractScraper()
        url = match_data.get('url') if isinstance(match_data, dict) else match_data.iloc[0].get('url')
        print("Fetching raw pointlog data for momentum learning...")
        raw_points = scraper.get_raw_pointlog(url)
        self.simulation_model.state_modifiers.fit_momentum(raw_points)

        # Cross-validation for ensemble weights
        self._optimize_ensemble_weights(match_data)

    def _optimize_ensemble_weights(self, match_data: pd.DataFrame):
        """Find optimal ensemble weights via cross-validation"""
        tscv = TimeSeriesSplit(n_splits=5)
        best_weights = None
        best_score = float('inf')

        for w_sim in np.arange(0.3, 0.8, 0.1):
            w_direct = 1 - w_sim
            scores = []

            for train_idx, val_idx in tscv.split(match_data):
                val_data = match_data.iloc[val_idx]

                # Get predictions (simplified - would run actual simulation)
                sim_probs = val_data['simulation_prob']  # Pre-computed
                direct_probs = val_data['direct_prob']  # Pre-computed

                ensemble_probs = w_sim * sim_probs + w_direct * direct_probs
                y_true = (val_data['actual_winner'] == 1).astype(int)

                score = brier_score_loss(y_true, ensemble_probs)
                scores.append(score)

            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_weights = {'simulation': w_sim, 'direct': w_direct}

        self.match_ensemble.ensemble_weights = best_weights
        print(f"Optimal weights: {best_weights}")

    def predict(self, match_context: dict, best_of: Optional[int] = None) -> dict:
        """Make prediction for a match"""
        # Determine best_of parameter: use argument, or match_context, or default to 3
        bo = best_of or match_context.get('best_of', 3)
        # Run simulation
        sim_prob = self.simulation_model.simulate_match(match_context, best_of=bo)

        # Get direct prediction
        match_features = pd.DataFrame([match_context])
        ensemble_prob = self.match_ensemble.predict(match_features, sim_prob)

        return {
            'win_probability': ensemble_prob,
            'simulation_component': sim_prob,
            'direct_component': ensemble_prob,  # Would separate these
            'confidence': self._calculate_confidence(ensemble_prob, match_context)
        }

    def _calculate_confidence(self, prob: float, context: dict) -> str:
        """Assess prediction confidence"""
        # Based on probability extremity and data quality
        extremity = abs(prob - 0.5)
        data_quality = context.get('data_quality_score', 0.5)

        confidence_score = extremity * data_quality

        if confidence_score > 0.15:
            return 'HIGH'
        elif confidence_score > 0.08:
            return 'MEDIUM'
        else:
            return 'LOW'

    def save(self, path: str):
        """Save trained models"""
        joblib.dump({
            'point_model': self.point_model,
            'match_ensemble': self.match_ensemble,
            'ensemble_weights': self.match_ensemble.ensemble_weights
        }, path)

    def load(self, path: str):
        """Load trained models"""
        components = joblib.load(path)
        self.point_model = components['point_model']
        self.match_ensemble = components['match_ensemble']
        self.simulation_model = DataDrivenTennisModel(self.point_model)