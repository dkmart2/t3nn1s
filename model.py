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
from sklearn.ensemble import StackingClassifier
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
from sklearn.ensemble import StackingClassifier


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
        self.calibrator = None

    def engineer_point_features(self, point_data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from raw point data"""
        features = pd.DataFrame()

        # Basic serve features
        features['is_first_serve'] = point_data.get('2nd', pd.Series([None] * len(point_data))).isna().astype(int)
        features['serve_direction_wide'] = point_data.get('1st', pd.Series([''] * len(point_data))).str.contains('w',
                                                                                                                 na=False).astype(
            int)
        features['serve_direction_body'] = point_data.get('1st', pd.Series([''] * len(point_data))).str.contains('b',
                                                                                                                 na=False).astype(
            int)
        features['serve_direction_t'] = point_data.get('1st', pd.Series([''] * len(point_data))).str.contains('t',
                                                                                                              na=False).astype(
            int)

        # Rally features
        features['rally_length'] = point_data.get('rallyCount', pd.Series([1] * len(point_data))).fillna(1)
        features['is_net_point'] = (
                    point_data.get('1st', pd.Series([''] * len(point_data))).str.contains('@', na=False) |
                    point_data.get('2nd', pd.Series([''] * len(point_data))).str.contains('@', na=False)).astype(int)

        # Enhanced rally features
        features['is_long_rally'] = (features['rally_length'] > 7).astype(int)
        features['is_short_point'] = (features['rally_length'] <= 3).astype(int)

        # Shot type features
        features['has_volley'] = (point_data.get('1st', pd.Series([''] * len(point_data))).str.contains('v', na=False) |
                                  point_data.get('2nd', pd.Series([''] * len(point_data))).str.contains('v',
                                                                                                        na=False)).astype(
            int)
        features['has_dropshot'] = (
                    point_data.get('1st', pd.Series([''] * len(point_data))).str.contains('d', na=False) |
                    point_data.get('2nd', pd.Series([''] * len(point_data))).str.contains('d', na=False)).astype(int)

        # Score state
        features['games_diff'] = point_data.get('Gm1', 0) - point_data.get('Gm2', 0)
        features['sets_diff'] = point_data.get('Set1', 0) - point_data.get('Set2', 0)
        features['is_tiebreak'] = point_data.get('TbSet', False).astype(int)

        # Point importance - Enhanced
        point_score = point_data.get('Pts', '0-0').fillna('0-0')
        features['is_break_point'] = ((point_score.isin(['0-40', '15-40', '30-40', 'AD-40'])) &
                                      (point_data.get('Svr', 1) == 2)).astype(int)
        features['is_game_point'] = point_score.isin(['40-0', '40-15', '40-30', '40-AD']).astype(int)
        features['is_deuce'] = point_score.isin(['40-40', 'deuce']).astype(int)
        features['is_set_point'] = ((features['games_diff'] >= 1) & (point_data.get('Gm1', 0) >= 5) |
                                    (features['games_diff'] <= -1) & (point_data.get('Gm2', 0) >= 5)).astype(int)

        # Surface
        surface = point_data.get('surface', 'Hard').fillna('Hard')
        features['surface_clay'] = (surface == 'Clay').astype(int)
        features['surface_grass'] = (surface == 'Grass').astype(int)
        features['surface_hard'] = (surface == 'Hard').astype(int)

        # Player strength differential
        features['elo_diff'] = point_data.get('server_elo', 1500) - point_data.get('returner_elo', 1500)
        features['rank_diff'] = point_data.get('returner_rank', 50) - point_data.get('server_rank', 50)
        features['h2h_server_advantage'] = point_data.get('server_h2h_win_pct', 0.5) - 0.5

        # Contextual flags
        features['is_grand_slam'] = point_data.get('tournament_tier', '').str.contains('Grand Slam', na=False).astype(
            int)
        features['is_masters'] = point_data.get('tournament_tier', '').str.contains('Masters', na=False).astype(int)
        features['round_level'] = pd.Categorical(point_data.get('round', 'R1'),
                                                 categories=['R1', 'R2', 'R3', 'R4', 'QF', 'SF', 'F']).codes

        # Fill NaN values
        features = features.fillna(0)
        self.feature_names = features.columns.tolist()
        return features

    def fit(self, point_data: pd.DataFrame):
        """Train the point-level model with proper calibration"""
        X = self.engineer_point_features(point_data)
        y = (point_data['PtWinner'] == point_data['Svr']).astype(int)

        # Remove NaN rows
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]

        if len(X) == 0:
            raise ValueError("No valid training data after filtering")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split for calibration
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_cal = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_cal = y[:split_idx], y[split_idx:]

        # Fit base model
        self.model.fit(X_train, y_train)

        # Calibrate with isotonic regression
        self.calibrator = CalibratedClassifierCV(
            self.model, method='isotonic', cv='prefit'
        )
        self.calibrator.fit(X_cal, y_cal)

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        return None

    def predict_proba(self, point_features: pd.DataFrame) -> np.ndarray:
        """Predict calibrated point-win probability"""
        X_scaled = self.scaler.transform(point_features)
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X_scaled)[:, 1]
        else:
            return self.model.predict_proba(X_scaled)[:, 1]


class StateDependentModifiers:
    """Momentum and pressure adjustments learned from data"""

    def __init__(self):
        self.momentum_decay = 0.85
        self.pressure_multipliers = {'server': {}, 'returner': {}}

    def calculate_momentum(self, recent_points: list, player: int) -> float:
        """Calculate momentum based on recent point outcomes"""
        if not recent_points:
            return 0.0

        weights = np.array([self.momentum_decay ** i for i in range(len(recent_points))])
        player_wins = np.array([1 if p == player else -1 for p in recent_points])

        momentum = np.sum(weights * player_wins) / np.sum(weights)
        return np.tanh(momentum * 0.3)

    def get_pressure_modifier(self, score_state: dict, player_role: str = 'server') -> float:
        """Get pressure modifier based on score state and player role"""
        if score_state.get('is_match_point'):
            return self.pressure_multipliers[player_role].get('match_point', 1.0)
        elif score_state.get('is_set_point'):
            return self.pressure_multipliers[player_role].get('set_point', 1.0)
        elif score_state.get('is_break_point'):
            return self.pressure_multipliers[player_role].get('break_point', 1.0)
        return 1.0

    def fit(self, point_data: pd.DataFrame):
        """Learn pressure multipliers from historical point data"""
        if 'PtWinner' not in point_data.columns or 'Svr' not in point_data.columns:
            print("fit: missing PtWinner/Svr columns, using defaults")
            self.pressure_multipliers = {
                'server': {'break_point': 0.9, 'set_point': 1.0, 'match_point': 1.1},
                'returner': {'break_point': 1.1, 'set_point': 1.0, 'match_point': 0.9}
            }
            return

        # Overall server win probability
        overall_server_win = (point_data['PtWinner'] == point_data['Svr']).mean()

        if overall_server_win == 0:
            print("fit: no server wins found, using defaults")
            self.pressure_multipliers = {
                'server': {'break_point': 0.9, 'set_point': 1.0, 'match_point': 1.1},
                'returner': {'break_point': 1.1, 'set_point': 1.0, 'match_point': 0.9}
            }
            return

        # Learn pressure effects for server and returner separately
        for pressure_type in ['break_point', 'set_point', 'match_point']:
            col_name = f'is_{pressure_type}'
            if col_name in point_data.columns:
                pressure_mask = point_data[col_name] == 1
                if pressure_mask.any():
                    # Server perspective
                    server_pressure_win = (
                            point_data[pressure_mask]['PtWinner'] == point_data[pressure_mask]['Svr']
                    ).mean()
                    self.pressure_multipliers['server'][pressure_type] = (
                        server_pressure_win / overall_server_win if overall_server_win > 0 else 1.0
                    )

                    # Returner perspective (opposite)
                    returner_pressure_win = 1 - server_pressure_win
                    overall_returner_win = 1 - overall_server_win
                    self.pressure_multipliers['returner'][pressure_type] = (
                        returner_pressure_win / overall_returner_win if overall_returner_win > 0 else 1.0
                    )
                else:
                    self.pressure_multipliers['server'][pressure_type] = 1.0
                    self.pressure_multipliers['returner'][pressure_type] = 1.0
            else:
                # Default pressure multipliers when no data available
                defaults = {
                    'break_point': {'server': 0.9, 'returner': 1.1},
                    'set_point': {'server': 1.0, 'returner': 1.0},
                    'match_point': {'server': 1.1, 'returner': 0.9}
                }
                self.pressure_multipliers['server'][pressure_type] = defaults[pressure_type]['server']
                self.pressure_multipliers['returner'][pressure_type] = defaults[pressure_type]['returner']

    def fit_momentum(self, point_data: pd.DataFrame):
        """Learn momentum decay parameter from point sequences"""
        if 'Svr' not in point_data.columns or 'PtWinner' not in point_data.columns:
            print("fit_momentum: missing Svr/PtWinner columns, using default decay")
            return

        if 'match_id' not in point_data.columns:
            print("fit_momentum: missing match_id, treating all points as one match")
            point_data = point_data.copy()
            point_data['match_id'] = 'default_match'

        best_decay = self.momentum_decay
        best_corr = -float('inf')

        # Test different decay values
        for decay in np.linspace(0.5, 0.99, 10):
            correlations = []

            # Group by match to calculate momentum within each match
            for match_id, match_points in point_data.groupby('match_id'):
                if len(match_points) < 5:  # Skip very short matches
                    continue

                match_points = match_points.sort_values('Pt' if 'Pt' in match_points.columns else match_points.index)
                momentums = []
                outcomes = []

                for i in range(len(match_points)):
                    server = match_points.iloc[i]['Svr']
                    winner = match_points.iloc[i]['PtWinner']
                    outcomes.append(1 if winner == server else 0)

                    # Calculate momentum based on previous points in this match
                    if i > 0:
                        prev_winners = match_points.iloc[:i]['PtWinner'].tolist()
                        weights = np.array([decay ** (i - k - 1) for k in range(i)])
                        signs = np.array([1 if w == server else -1 for w in prev_winners])
                        momentum = (weights * signs).sum() / weights.sum() if weights.sum() > 0 else 0
                    else:
                        momentum = 0.0
                    momentums.append(momentum)

                # Calculate correlation for this match
                if len(momentums) > 1 and np.var(momentums) > 0:
                    corr = np.corrcoef(momentums, outcomes)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

            # Average correlation across all matches
            if correlations:
                avg_corr = np.mean(correlations)
                if avg_corr > best_corr:
                    best_corr = avg_corr
                    best_decay = decay

        self.momentum_decay = best_decay
        print(f"Learned momentum_decay = {self.momentum_decay:.3f} (corr={best_corr:.3f})")


class DataDrivenTennisModel:
    """Enhanced model with ML-based point probabilities"""

    def __init__(self, point_model: PointLevelModel, n_simulations: int = 1000):
        self.point_model = point_model
        self.n_simulations = n_simulations
        self.state_modifiers = StateDependentModifiers()
        self.recent_points = []

    def get_point_win_prob(self, match_context: dict, score_state: dict, momentum: dict) -> float:
        """Get point-win probability from trained model"""
        # Build feature vector
        features = pd.DataFrame([{
            'is_first_serve': 1,
            'serve_direction_wide': 0.3,
            'serve_direction_body': 0.3,
            'serve_direction_t': 0.4,
            'rally_length': 4.5,
            'is_net_point': 0,
            'is_long_rally': 0,
            'is_short_point': 0,
            'has_volley': 0,
            'has_dropshot': 0,
            'games_diff': score_state.get('games_diff', 0),
            'sets_diff': score_state.get('sets_diff', 0),
            'is_tiebreak': score_state.get('is_tiebreak', 0),
            'is_break_point': score_state.get('is_break_point', 0),
            'is_game_point': score_state.get('is_game_point', 0),
            'is_deuce': score_state.get('is_deuce', 0),
            'is_set_point': score_state.get('is_set_point', 0),
            'surface_clay': match_context.get('surface') == 'Clay',
            'surface_grass': match_context.get('surface') == 'Grass',
            'surface_hard': match_context.get('surface') == 'Hard',
            'elo_diff': match_context.get('elo_diff', 0),
            'rank_diff': match_context.get('rank_diff', 0),
            'h2h_server_advantage': match_context.get('h2h_advantage', 0),
            'is_grand_slam': match_context.get('is_grand_slam', 0),
            'is_masters': match_context.get('is_masters', 0),
            'round_level': match_context.get('round_level', 0)
        }])

        # Get base probability
        base_prob = self.point_model.predict_proba(features)[0]

        # Apply modifiers
        server_pressure = self.state_modifiers.get_pressure_modifier(score_state, 'server')
        momentum_mod = 1 + momentum.get('server', 0) * 0.05

        adjusted_prob = base_prob * server_pressure * momentum_mod
        return np.clip(adjusted_prob, 0.01, 0.99)

    def simulate_game(self, match_context: dict, score_state: dict, server: int) -> int:
        """Simulate game with dynamic probabilities"""
        points = {'server': 0, 'returner': 0}
        momentum = {'server': 0, 'returner': 0}

        while True:
            momentum['server'] = self.state_modifiers.calculate_momentum(
                self.recent_points[-10:], server
            )
            momentum['returner'] = -momentum['server']

            score_state.update({
                'is_break_point': (points['returner'] >= 3 and points['returner'] > points['server']),
                'is_game_point': (points['server'] >= 3 and points['server'] > points['returner']),
                'is_deuce': (points['server'] >= 3 and points['returner'] >= 3 and
                             abs(points['server'] - points['returner']) < 2)
            })

            point_prob = self.get_point_win_prob(match_context, score_state, momentum)

            if np.random.random() < point_prob:
                points['server'] += 1
                self.recent_points.append(server)
            else:
                points['returner'] += 1
                self.recent_points.append(3 - server)

            if (points['server'] >= 4 or points['returner'] >= 4) and \
                    abs(points['server'] - points['returner']) >= 2:
                return server if points['server'] > points['returner'] else 3 - server

    def simulate_match(self, match_context: dict, best_of: int = 3) -> float:
        """Run Monte Carlo simulation"""
        wins = 0
        sets_to_win = best_of // 2 + 1

        for _ in range(self.n_simulations):
            self.recent_points = []
            p1_sets = p2_sets = 0

            while p1_sets < sets_to_win and p2_sets < sets_to_win:
                set_winner = self._simulate_set(match_context, p1_sets, p2_sets)
                if set_winner == 1:
                    p1_sets += 1
                else:
                    p2_sets += 1

            if p1_sets > p2_sets:
                wins += 1

        return wins / self.n_simulations

    def _simulate_set(self, match_context: dict, p1_sets: int, p2_sets: int) -> int:
        """Simulate a set"""
        p1_games = p2_games = 0
        server = 1

        while True:
            score_state = {
                'games_diff': p1_games - p2_games,
                'sets_diff': p1_sets - p2_sets,
                'is_tiebreak': False,
                'is_set_point': (p1_games >= 5 or p2_games >= 5) and abs(p1_games - p2_games) >= 1
            }

            game_winner = self.simulate_game(match_context, score_state, server)

            if game_winner == 1:
                p1_games += 1
            else:
                p2_games += 1

            server = 3 - server

            if (p1_games >= 6 or p2_games >= 6) and abs(p1_games - p2_games) >= 2:
                return 1 if p1_games > p2_games else 2
            elif p1_games == 6 and p2_games == 6:
                return self._simulate_tiebreak(match_context)

    def _simulate_tiebreak(self, match_context: dict) -> int:
        """Simulate tiebreak"""
        score_state = {'is_tiebreak': True}
        momentum = {'server': 0}
        prob = self.get_point_win_prob(match_context, score_state, momentum)
        return 1 if np.random.random() < prob else 2


class MatchLevelEnsemble:
    """Direct match prediction + simulation ensemble with stacking"""

    def __init__(self):
        # Base models for stacking
        self.lgb_model = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.03)
        self.rf_model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)

        # Meta-learner for stacking
        self.stacking_model = StackingClassifier(
            estimators=[
                ('lgb', self.lgb_model),
                ('rf', self.rf_model)
            ],
            final_estimator=LogisticRegression(),
            cv=TimeSeriesSplit(n_splits=3)
        )

        self.calibrator = IsotonicRegression(out_of_bounds='clip')

    def engineer_match_features(self, match_data: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced match-level features"""
        features = pd.DataFrame()

        # Basic features
        features['rank_diff'] = match_data.get('WRank', 50) - match_data.get('LRank', 50)
        features['elo_diff'] = match_data.get('winner_elo', 1500) - match_data.get('loser_elo', 1500)
        features['h2h_balance'] = match_data.get('p1_h2h_win_pct', 0.5) - 0.5

        # Surface-specific H2H
        features['h2h_surface_diff'] = (match_data.get('p1_surface_h2h_wins', 0) -
                                        match_data.get('p2_surface_h2h_wins', 0))

        # Serve stats
        features['winner_serve_dominance'] = (
                match_data.get('winner_aces', 0) / match_data.get('winner_serve_pts', 1).clip(lower=1)
        )
        features['loser_serve_dominance'] = (
                match_data.get('loser_aces', 0) / match_data.get('loser_serve_pts', 1).clip(lower=1)
        )

        # Form indicators
        features['winner_recent_win_pct'] = match_data.get('winner_last10_wins', 5) / 10
        features['loser_recent_win_pct'] = match_data.get('loser_last10_wins', 5) / 10

        # Tournament context
        features['is_grand_slam'] = match_data.get('tournament_tier', '').str.contains('Grand Slam', na=False).astype(
            int)
        features['is_masters'] = match_data.get('tournament_tier', '').str.contains('Masters', na=False).astype(int)

        # Round level
        round_mapping = {'R1': 1, 'R2': 2, 'R3': 3, 'R4': 4, 'QF': 5, 'SF': 6, 'F': 7}
        features['round_level'] = match_data.get('Round', 'R1').map(round_mapping).fillna(1)

        # Days since last match (contextual)
        features['days_since_last_match'] = match_data.get('days_since_last_match', 7)

        return features.fillna(0)

    def fit(self, historical_data: pd.DataFrame):
        """Train ensemble with stacking"""
        X = self.engineer_match_features(historical_data)
        y = (historical_data.get('actual_winner', historical_data.get('Winner')) ==
             historical_data.get('winner_canonical', historical_data.get('Winner'))).astype(int)

        # Remove NaN
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]

        if len(X) == 0:
            raise ValueError("No valid training data")

        # Time-based split for calibration
        split_idx = int(len(X) * 0.8)
        X_train, X_cal = X[:split_idx], X[split_idx:]
        y_train, y_cal = y[:split_idx], y[split_idx:]

        # Train stacking ensemble
        self.stacking_model.fit(X_train, y_train)

        # Calibrate
        cal_probs = self.stacking_model.predict_proba(X_cal)[:, 1]
        self.calibrator.fit(cal_probs, y_cal)

    def predict(self, match_features: pd.DataFrame, simulation_prob: float) -> float:
        """Ensemble prediction with simulation input"""
        # Get direct prediction
        direct_prob = self.stacking_model.predict_proba(match_features)[:, 1][0]
        direct_prob_cal = self.calibrator.transform([direct_prob])[0]

        # Weighted combination (can be learned)
        final_prob = 0.6 * simulation_prob + 0.4 * direct_prob_cal
        return np.clip(final_prob, 0.01, 0.99)


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
        if feature_importance is not None:
            print(f"Top features:\n{feature_importance.head(10)}")

        print("\nTraining match-level ensemble...")
        self.match_ensemble.fit(match_data)

        print("\nInitializing simulation model...")
        self.simulation_model = DataDrivenTennisModel(self.point_model)
        self.simulation_model.state_modifiers.fit(point_data)
        self.simulation_model.state_modifiers.fit_momentum(point_data)

    def predict(self, match_context: dict, best_of: Optional[int] = None) -> dict:
        """Make prediction for a match"""
        bo = best_of or match_context.get('best_of', 3)

        # Run simulation
        sim_prob = self.simulation_model.simulate_match(match_context, best_of=bo)

        # Get direct prediction
        match_features = pd.DataFrame([match_context])
        ensemble_prob = self.match_ensemble.predict(match_features, sim_prob)

        return {
            'win_probability': ensemble_prob,
            'simulation_component': sim_prob,
            'direct_component': ensemble_prob,
            'confidence': self._calculate_confidence(ensemble_prob, match_context)
        }

    def _calculate_confidence(self, prob: float, context: dict) -> str:
        """Assess prediction confidence"""
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
            'simulation_model': self.simulation_model
        }, path)

    def load(self, path: str):
        """Load trained models"""
        components = joblib.load(path)
        self.point_model = components['point_model']
        self.match_ensemble = components['match_ensemble']
        if 'simulation_model' in components:
            self.simulation_model = components['simulation_model']
        else:
            self.simulation_model = DataDrivenTennisModel(self.point_model)

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
        self.calibrator = None

    def engineer_point_features(self, point_data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from raw point data"""
        features = pd.DataFrame()

        # Basic serve features
        features['is_first_serve'] = point_data.get('2nd', pd.Series([None] * len(point_data))).isna().astype(int)
        features['serve_direction_wide'] = point_data.get('1st', pd.Series([''] * len(point_data))).str.contains('w',
                                                                                                                 na=False).astype(
            int)
        features['serve_direction_body'] = point_data.get('1st', pd.Series([''] * len(point_data))).str.contains('b',
                                                                                                                 na=False).astype(
            int)
        features['serve_direction_t'] = point_data.get('1st', pd.Series([''] * len(point_data))).str.contains('t',
                                                                                                              na=False).astype(
            int)

        # Rally features
        features['rally_length'] = point_data.get('rallyCount', pd.Series([1] * len(point_data))).fillna(1)
        features['is_net_point'] = (
                    point_data.get('1st', pd.Series([''] * len(point_data))).str.contains('@', na=False) |
                    point_data.get('2nd', pd.Series([''] * len(point_data))).str.contains('@', na=False)).astype(int)

        # Enhanced rally features
        features['is_long_rally'] = (features['rally_length'] > 7).astype(int)
        features['is_short_point'] = (features['rally_length'] <= 3).astype(int)

        # Shot type features
        features['has_volley'] = (point_data.get('1st', pd.Series([''] * len(point_data))).str.contains('v', na=False) |
                                  point_data.get('2nd', pd.Series([''] * len(point_data))).str.contains('v',
                                                                                                        na=False)).astype(
            int)
        features['has_dropshot'] = (
                    point_data.get('1st', pd.Series([''] * len(point_data))).str.contains('d', na=False) |
                    point_data.get('2nd', pd.Series([''] * len(point_data))).str.contains('d', na=False)).astype(int)

        # Score state
        features['games_diff'] = point_data.get('Gm1', 0) - point_data.get('Gm2', 0)
        features['sets_diff'] = point_data.get('Set1', 0) - point_data.get('Set2', 0)
        features['is_tiebreak'] = point_data.get('TbSet', False).astype(int)

        # Point importance - Enhanced
        point_score = point_data.get('Pts', '0-0').fillna('0-0')
        features['is_break_point'] = ((point_score.isin(['0-40', '15-40', '30-40', 'AD-40'])) &
                                      (point_data.get('Svr', 1) == 2)).astype(int)
        features['is_game_point'] = point_score.isin(['40-0', '40-15', '40-30', '40-AD']).astype(int)
        features['is_deuce'] = point_score.isin(['40-40', 'deuce']).astype(int)
        features['is_set_point'] = ((features['games_diff'] >= 1) & (point_data.get('Gm1', 0) >= 5) |
                                    (features['games_diff'] <= -1) & (point_data.get('Gm2', 0) >= 5)).astype(int)

        # Surface
        surface = point_data.get('surface', 'Hard').fillna('Hard')
        features['surface_clay'] = (surface == 'Clay').astype(int)
        features['surface_grass'] = (surface == 'Grass').astype(int)
        features['surface_hard'] = (surface == 'Hard').astype(int)

        # Player strength differential
        features['elo_diff'] = point_data.get('server_elo', 1500) - point_data.get('returner_elo', 1500)
        features['rank_diff'] = point_data.get('returner_rank', 50) - point_data.get('server_rank', 50)
        features['h2h_server_advantage'] = point_data.get('server_h2h_win_pct', 0.5) - 0.5

        # Contextual flags
        features['is_grand_slam'] = point_data.get('tournament_tier', '').str.contains('Grand Slam', na=False).astype(
            int)
        features['is_masters'] = point_data.get('tournament_tier', '').str.contains('Masters', na=False).astype(int)
        features['round_level'] = pd.Categorical(point_data.get('round', 'R1'),
                                                 categories=['R1', 'R2', 'R3', 'R4', 'QF', 'SF', 'F']).codes

        # Fill NaN values
        features = features.fillna(0)
        self.feature_names = features.columns.tolist()
        return features

    def fit(self, point_data: pd.DataFrame):
        """Train the point-level model with proper calibration"""
        X = self.engineer_point_features(point_data)
        y = (point_data['PtWinner'] == point_data['Svr']).astype(int)

        # Remove NaN rows
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]

        if len(X) == 0:
            raise ValueError("No valid training data after filtering")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split for calibration
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_cal = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_cal = y[:split_idx], y[split_idx:]

        # Fit base model
        self.model.fit(X_train, y_train)

        # Calibrate with isotonic regression
        self.calibrator = CalibratedClassifierCV(
            self.model, method='isotonic', cv='prefit'
        )
        self.calibrator.fit(X_cal, y_cal)

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        return None

    def predict_proba(self, point_features: pd.DataFrame) -> np.ndarray:
        """Predict calibrated point-win probability"""
        X_scaled = self.scaler.transform(point_features)
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X_scaled)[:, 1]
        else:
            return self.model.predict_proba(X_scaled)[:, 1]


class StateDependentModifiers:
    """Momentum and pressure adjustments learned from data"""

    def __init__(self):
        self.momentum_decay = 0.85
        self.pressure_multipliers = {'server': {}, 'returner': {}}

    def calculate_momentum(self, recent_points: list, player: int) -> float:
        """Calculate momentum based on recent point outcomes"""
        if not recent_points:
            return 0.0

        weights = np.array([self.momentum_decay ** i for i in range(len(recent_points))])
        player_wins = np.array([1 if p == player else -1 for p in recent_points])

        momentum = np.sum(weights * player_wins) / np.sum(weights)
        return np.tanh(momentum * 0.3)

    def get_pressure_modifier(self, score_state: dict, player_role: str = 'server') -> float:
        """Get pressure modifier based on score state and player role"""
        if score_state.get('is_match_point'):
            return self.pressure_multipliers[player_role].get('match_point', 1.0)
        elif score_state.get('is_set_point'):
            return self.pressure_multipliers[player_role].get('set_point', 1.0)
        elif score_state.get('is_break_point'):
            return self.pressure_multipliers[player_role].get('break_point', 1.0)
        return 1.0

    def fit(self, point_data: pd.DataFrame):
        """Learn pressure multipliers from historical point data"""
        if 'PtWinner' not in point_data.columns or 'Svr' not in point_data.columns:
            print("fit: missing PtWinner/Svr columns, using defaults")
            self.pressure_multipliers = {
                'server': {'break_point': 0.9, 'set_point': 1.0, 'match_point': 1.1},
                'returner': {'break_point': 1.1, 'set_point': 1.0, 'match_point': 0.9}
            }
            return

        # Overall server win probability
        overall_server_win = (point_data['PtWinner'] == point_data['Svr']).mean()

        if overall_server_win == 0:
            print("fit: no server wins found, using defaults")
            self.pressure_multipliers = {
                'server': {'break_point': 0.9, 'set_point': 1.0, 'match_point': 1.1},
                'returner': {'break_point': 1.1, 'set_point': 1.0, 'match_point': 0.9}
            }
            return

        # Learn pressure effects for server and returner separately
        for pressure_type in ['break_point', 'set_point', 'match_point']:
            col_name = f'is_{pressure_type}'
            if col_name in point_data.columns:
                pressure_mask = point_data[col_name] == 1
                if pressure_mask.any():
                    # Server perspective
                    server_pressure_win = (
                            point_data[pressure_mask]['PtWinner'] == point_data[pressure_mask]['Svr']
                    ).mean()
                    self.pressure_multipliers['server'][pressure_type] = (
                        server_pressure_win / overall_server_win if overall_server_win > 0 else 1.0
                    )

                    # Returner perspective (opposite)
                    returner_pressure_win = 1 - server_pressure_win
                    overall_returner_win = 1 - overall_server_win
                    self.pressure_multipliers['returner'][pressure_type] = (
                        returner_pressure_win / overall_returner_win if overall_returner_win > 0 else 1.0
                    )
                else:
                    self.pressure_multipliers['server'][pressure_type] = 1.0
                    self.pressure_multipliers['returner'][pressure_type] = 1.0
            else:
                # Default pressure multipliers when no data available
                defaults = {
                    'break_point': {'server': 0.9, 'returner': 1.1},
                    'set_point': {'server': 1.0, 'returner': 1.0},
                    'match_point': {'server': 1.1, 'returner': 0.9}
                }
                self.pressure_multipliers['server'][pressure_type] = defaults[pressure_type]['server']
                self.pressure_multipliers['returner'][pressure_type] = defaults[pressure_type]['returner']

    def fit_momentum(self, point_data: pd.DataFrame):
        """Learn momentum decay parameter from point sequences"""
        if 'Svr' not in point_data.columns or 'PtWinner' not in point_data.columns:
            print("fit_momentum: missing Svr/PtWinner columns, using default decay")
            return

        if 'match_id' not in point_data.columns:
            print("fit_momentum: missing match_id, treating all points as one match")
            point_data = point_data.copy()
            point_data['match_id'] = 'default_match'

        best_decay = self.momentum_decay
        best_corr = -float('inf')

        # Test different decay values
        for decay in np.linspace(0.5, 0.99, 10):
            correlations = []

            # Group by match to calculate momentum within each match
            for match_id, match_points in point_data.groupby('match_id'):
                if len(match_points) < 5:  # Skip very short matches
                    continue

                match_points = match_points.sort_values('Pt' if 'Pt' in match_points.columns else match_points.index)
                momentums = []
                outcomes = []

                for i in range(len(match_points)):
                    server = match_points.iloc[i]['Svr']
                    winner = match_points.iloc[i]['PtWinner']
                    outcomes.append(1 if winner == server else 0)

                    # Calculate momentum based on previous points in this match
                    if i > 0:
                        prev_winners = match_points.iloc[:i]['PtWinner'].tolist()
                        weights = np.array([decay ** (i - k - 1) for k in range(i)])
                        signs = np.array([1 if w == server else -1 for w in prev_winners])
                        momentum = (weights * signs).sum() / weights.sum() if weights.sum() > 0 else 0
                    else:
                        momentum = 0.0
                    momentums.append(momentum)

                # Calculate correlation for this match
                if len(momentums) > 1 and np.var(momentums) > 0:
                    corr = np.corrcoef(momentums, outcomes)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

            # Average correlation across all matches
            if correlations:
                avg_corr = np.mean(correlations)
                if avg_corr > best_corr:
                    best_corr = avg_corr
                    best_decay = decay

        self.momentum_decay = best_decay
        print(f"Learned momentum_decay = {self.momentum_decay:.3f} (corr={best_corr:.3f})")


class DataDrivenTennisModel:
    """Enhanced model with ML-based point probabilities"""

    def __init__(self, point_model: PointLevelModel, n_simulations: int = 1000):
        self.point_model = point_model
        self.n_simulations = n_simulations
        self.state_modifiers = StateDependentModifiers()
        self.recent_points = []

    def get_point_win_prob(self, match_context: dict, score_state: dict, momentum: dict) -> float:
        """Get point-win probability from trained model"""
        # Build feature vector
        features = pd.DataFrame([{
            'is_first_serve': 1,
            'serve_direction_wide': 0.3,
            'serve_direction_body': 0.3,
            'serve_direction_t': 0.4,
            'rally_length': 4.5,
            'is_net_point': 0,
            'is_long_rally': 0,
            'is_short_point': 0,
            'has_volley': 0,
            'has_dropshot': 0,
            'games_diff': score_state.get('games_diff', 0),
            'sets_diff': score_state.get('sets_diff', 0),
            'is_tiebreak': score_state.get('is_tiebreak', 0),
            'is_break_point': score_state.get('is_break_point', 0),
            'is_game_point': score_state.get('is_game_point', 0),
            'is_deuce': score_state.get('is_deuce', 0),
            'is_set_point': score_state.get('is_set_point', 0),
            'surface_clay': match_context.get('surface') == 'Clay',
            'surface_grass': match_context.get('surface') == 'Grass',
            'surface_hard': match_context.get('surface') == 'Hard',
            'elo_diff': match_context.get('elo_diff', 0),
            'rank_diff': match_context.get('rank_diff', 0),
            'h2h_server_advantage': match_context.get('h2h_advantage', 0),
            'is_grand_slam': match_context.get('is_grand_slam', 0),
            'is_masters': match_context.get('is_masters', 0),
            'round_level': match_context.get('round_level', 0)
        }])

        # Get base probability
        base_prob = self.point_model.predict_proba(features)[0]

        # Apply modifiers
        server_pressure = self.state_modifiers.get_pressure_modifier(score_state, 'server')
        momentum_mod = 1 + momentum.get('server', 0) * 0.05

        adjusted_prob = base_prob * server_pressure * momentum_mod
        return np.clip(adjusted_prob, 0.01, 0.99)

    def simulate_game(self, match_context: dict, score_state: dict, server: int) -> int:
        """Simulate game with dynamic probabilities"""
        points = {'server': 0, 'returner': 0}
        momentum = {'server': 0, 'returner': 0}

        while True:
            momentum['server'] = self.state_modifiers.calculate_momentum(
                self.recent_points[-10:], server
            )
            momentum['returner'] = -momentum['server']

            score_state.update({
                'is_break_point': (points['returner'] >= 3 and points['returner'] > points['server']),
                'is_game_point': (points['server'] >= 3 and points['server'] > points['returner']),
                'is_deuce': (points['server'] >= 3 and points['returner'] >= 3 and
                             abs(points['server'] - points['returner']) < 2)
            })

            point_prob = self.get_point_win_prob(match_context, score_state, momentum)

            if np.random.random() < point_prob:
                points['server'] += 1
                self.recent_points.append(server)
            else:
                points['returner'] += 1
                self.recent_points.append(3 - server)

            if (points['server'] >= 4 or points['returner'] >= 4) and \
                    abs(points['server'] - points['returner']) >= 2:
                return server if points['server'] > points['returner'] else 3 - server

    def simulate_match(self, match_context: dict, best_of: int = 3) -> float:
        """Run Monte Carlo simulation"""
        wins = 0
        sets_to_win = best_of // 2 + 1

        for _ in range(self.n_simulations):
            self.recent_points = []
            p1_sets = p2_sets = 0

            while p1_sets < sets_to_win and p2_sets < sets_to_win:
                set_winner = self._simulate_set(match_context, p1_sets, p2_sets)
                if set_winner == 1:
                    p1_sets += 1
                else:
                    p2_sets += 1

            if p1_sets > p2_sets:
                wins += 1

        return wins / self.n_simulations

    def _simulate_set(self, match_context: dict, p1_sets: int, p2_sets: int) -> int:
        """Simulate a set"""
        p1_games = p2_games = 0
        server = 1

        while True:
            score_state = {
                'games_diff': p1_games - p2_games,
                'sets_diff': p1_sets - p2_sets,
                'is_tiebreak': False,
                'is_set_point': (p1_games >= 5 or p2_games >= 5) and abs(p1_games - p2_games) >= 1
            }

            game_winner = self.simulate_game(match_context, score_state, server)

            if game_winner == 1:
                p1_games += 1
            else:
                p2_games += 1

            server = 3 - server

            if (p1_games >= 6 or p2_games >= 6) and abs(p1_games - p2_games) >= 2:
                return 1 if p1_games > p2_games else 2
            elif p1_games == 6 and p2_games == 6:
                return self._simulate_tiebreak(match_context)

    def _simulate_tiebreak(self, match_context: dict) -> int:
        """Simulate tiebreak"""
        score_state = {'is_tiebreak': True}
        momentum = {'server': 0}
        prob = self.get_point_win_prob(match_context, score_state, momentum)
        return 1 if np.random.random() < prob else 2


class MatchLevelEnsemble:
    """Direct match prediction + simulation ensemble with stacking"""

    def __init__(self):
        # Base models for stacking
        self.lgb_model = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.03)
        self.rf_model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)

        # Meta-learner for stacking
        self.stacking_model = StackingClassifier(
            estimators=[
                ('lgb', self.lgb_model),
                ('rf', self.rf_model)
            ],
            final_estimator=LogisticRegression(),
            cv=TimeSeriesSplit(n_splits=3)
        )

        self.calibrator = IsotonicRegression(out_of_bounds='clip')

    def engineer_match_features(self, match_data: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced match-level features"""
        features = pd.DataFrame()

        # Basic features
        features['rank_diff'] = match_data.get('WRank', 50) - match_data.get('LRank', 50)
        features['elo_diff'] = match_data.get('winner_elo', 1500) - match_data.get('loser_elo', 1500)
        features['h2h_balance'] = match_data.get('p1_h2h_win_pct', 0.5) - 0.5

        # Surface-specific H2H
        features['h2h_surface_diff'] = (match_data.get('p1_surface_h2h_wins', 0) -
                                        match_data.get('p2_surface_h2h_wins', 0))

        # Serve stats
        features['winner_serve_dominance'] = (
                match_data.get('winner_aces', 0) / match_data.get('winner_serve_pts', 1).clip(lower=1)
        )
        features['loser_serve_dominance'] = (
                match_data.get('loser_aces', 0) / match_data.get('loser_serve_pts', 1).clip(lower=1)
        )

        # Form indicators
        features['winner_recent_win_pct'] = match_data.get('winner_last10_wins', 5) / 10
        features['loser_recent_win_pct'] = match_data.get('loser_last10_wins', 5) / 10

        # Tournament context
        features['is_grand_slam'] = match_data.get('tournament_tier', '').str.contains('Grand Slam', na=False).astype(
            int)
        features['is_masters'] = match_data.get('tournament_tier', '').str.contains('Masters', na=False).astype(int)

        # Round level
        round_mapping = {'R1': 1, 'R2': 2, 'R3': 3, 'R4': 4, 'QF': 5, 'SF': 6, 'F': 7}
        features['round_level'] = match_data.get('Round', 'R1').map(round_mapping).fillna(1)

        # Days since last match (contextual)
        features['days_since_last_match'] = match_data.get('days_since_last_match', 7)

        return features.fillna(0)

    def fit(self, historical_data: pd.DataFrame):
        """Train ensemble with stacking"""
        X = self.engineer_match_features(historical_data)
        y = (historical_data.get('actual_winner', historical_data.get('Winner')) ==
             historical_data.get('winner_canonical', historical_data.get('Winner'))).astype(int)

        # Remove NaN
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]

        if len(X) == 0:
            raise ValueError("No valid training data")

        # Time-based split for calibration
        split_idx = int(len(X) * 0.8)
        X_train, X_cal = X[:split_idx], X[split_idx:]
        y_train, y_cal = y[:split_idx], y[split_idx:]

        # Train stacking ensemble
        self.stacking_model.fit(X_train, y_train)

        # Calibrate
        cal_probs = self.stacking_model.predict_proba(X_cal)[:, 1]
        self.calibrator.fit(cal_probs, y_cal)

    def predict(self, match_features: pd.DataFrame, simulation_prob: float) -> float:
        """Ensemble prediction with simulation input"""
        # Get direct prediction
        direct_prob = self.stacking_model.predict_proba(match_features)[:, 1][0]
        direct_prob_cal = self.calibrator.transform([direct_prob])[0]

        # Weighted combination (can be learned)
        final_prob = 0.6 * simulation_prob + 0.4 * direct_prob_cal
        return np.clip(final_prob, 0.01, 0.99)


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
        if feature_importance is not None:
            print(f"Top features:\n{feature_importance.head(10)}")

        print("\nTraining match-level ensemble...")
        self.match_ensemble.fit(match_data)

        print("\nInitializing simulation model...")
        self.simulation_model = DataDrivenTennisModel(self.point_model)
        self.simulation_model.state_modifiers.fit(point_data)
        self.simulation_model.state_modifiers.fit_momentum(point_data)

    def predict(self, match_context: dict, best_of: Optional[int] = None) -> dict:
        """Make prediction for a match"""
        bo = best_of or match_context.get('best_of', 3)

        # Run simulation
        sim_prob = self.simulation_model.simulate_match(match_context, best_of=bo)

        # Get direct prediction
        match_features = pd.DataFrame([match_context])
        ensemble_prob = self.match_ensemble.predict(match_features, sim_prob)

        return {
            'win_probability': ensemble_prob,
            'simulation_component': sim_prob,
            'direct_component': ensemble_prob,
            'confidence': self._calculate_confidence(ensemble_prob, match_context)
        }

    def _calculate_confidence(self, prob: float, context: dict) -> str:
        """Assess prediction confidence"""
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
            'simulation_model': self.simulation_model
        }, path)

    def load(self, path: str):
        """Load trained models"""
        components = joblib.load(path)
        self.point_model = components['point_model']
        self.match_ensemble = components['match_ensemble']
        if 'simulation_model' in components:
            self.simulation_model = components['simulation_model']
        else:
            self.simulation_model = DataDrivenTennisModel(self.point_model)