# Import-safe version of tennis_updated functions
import pandas as pd
import numpy as np
import random
from datetime import date
import os
import pickle

# Core functions only - no module-level execution

def load_from_cache():
    """Load data from cache"""
    CACHE_DIR = os.path.expanduser("~/Desktop/data/cache")
    HD_PATH = os.path.join(CACHE_DIR, "historical_data.parquet")
    JEFF_PATH = os.path.join(CACHE_DIR, "jeff_data.pkl")
    DEF_PATH = os.path.join(CACHE_DIR, "weighted_defaults.pkl")
    
    if (os.path.exists(HD_PATH) and os.path.exists(JEFF_PATH) and os.path.exists(DEF_PATH)):
        try:
            historical_data = pd.read_parquet(HD_PATH)
            with open(JEFF_PATH, "rb") as f:
                jeff_data = pickle.load(f)
            with open(DEF_PATH, "rb") as f:
                weighted_defaults = pickle.load(f)
            return historical_data, jeff_data, weighted_defaults
        except Exception as e:
            print(f"Cache load failed: {e}")
    return None, None, None

def extract_unified_features(match_data, player_prefix):
    """Extract features with API statistics support"""
    features = {}

    # Serve effectiveness (API > Tennis Abstract > Jeff > defaults)
    if f'{player_prefix}_points_service_points_won_pct' in match_data:
        features['serve_effectiveness'] = match_data[f'{player_prefix}_points_service_points_won_pct']
    elif f'{player_prefix}_ta_serve_won_pct' in match_data:
        features['serve_effectiveness'] = match_data[f'{player_prefix}_ta_serve_won_pct']
    elif f'{player_prefix}_serve_pts' in match_data and f'{player_prefix}_serve_pts' != 0:
        serve_pts = match_data.get(f'{player_prefix}_serve_pts', 80)
        first_won = match_data.get(f'{player_prefix}_first_won', 0)
        second_won = match_data.get(f'{player_prefix}_second_won', 0)
        features['serve_effectiveness'] = (first_won + second_won) / serve_pts if serve_pts > 0 else 0.65
    else:
        features['serve_effectiveness'] = 0.65

    # Return effectiveness
    if f'{player_prefix}_points_return_points_won_pct' in match_data:
        features['return_effectiveness'] = match_data[f'{player_prefix}_points_return_points_won_pct']
    elif f'{player_prefix}_ta_return_won_pct' in match_data:
        features['return_effectiveness'] = match_data[f'{player_prefix}_ta_return_won_pct']
    elif f'{player_prefix}_return_pts_won' in match_data:
        return_pts = match_data.get(f'{player_prefix}_return_pts', 80)
        return_won = match_data.get(f'{player_prefix}_return_pts_won', 25)
        features['return_effectiveness'] = return_won / return_pts if return_pts > 0 else 0.35
    else:
        features['return_effectiveness'] = 0.35

    # Winners rate
    if f'{player_prefix}_ta_overview_winners' in match_data:
        serve_pts = match_data.get(f'{player_prefix}_ta_serve_points', 80)
        return_pts = match_data.get(f'{player_prefix}_ta_return_points', 70)
        total_pts = serve_pts + return_pts
        features['winners_rate'] = match_data[f'{player_prefix}_ta_overview_winners'] / total_pts if total_pts > 0 else 0.20
    elif f'{player_prefix}_winners_total' in match_data:
        serve_pts = match_data.get(f'{player_prefix}_serve_pts', 80)
        features['winners_rate'] = match_data[f'{player_prefix}_winners_total'] / serve_pts if serve_pts > 0 else 0.20
    else:
        serve_eff = features.get('serve_effectiveness', 0.65)
        features['winners_rate'] = 0.1 + (serve_eff - 0.5) * 0.4

    # Unforced errors rate
    if f'{player_prefix}_ta_overview_unforced' in match_data:
        serve_pts = match_data.get(f'{player_prefix}_ta_serve_points', 80)
        return_pts = match_data.get(f'{player_prefix}_ta_return_points', 70) 
        total_pts = serve_pts + return_pts
        features['unforced_rate'] = match_data[f'{player_prefix}_ta_overview_unforced'] / total_pts if total_pts > 0 else 0.18
    elif f'{player_prefix}_unforced_errors' in match_data:
        serve_pts = match_data.get(f'{player_prefix}_serve_pts', 80)
        features['unforced_rate'] = match_data[f'{player_prefix}_unforced_errors'] / serve_pts if serve_pts > 0 else 0.18
    else:
        dfs = match_data.get(f'{player_prefix}_service_double_faults', 3)
        serve_total = match_data.get(f'{player_prefix}_points_service_points_won_total', 74)
        df_rate = dfs / serve_total if serve_total > 0 else 0.04
        features['unforced_rate'] = 0.12 + df_rate * 2

    # Pressure performance
    if f'{player_prefix}_pressure_performance' in match_data:
        features['pressure_performance'] = match_data[f'{player_prefix}_pressure_performance']
    elif f'{player_prefix}_key_points_serve_won_pct' in match_data:
        features['pressure_performance'] = match_data[f'{player_prefix}_key_points_serve_won_pct']
    elif f'{player_prefix}_service_break_points_saved_pct' in match_data:
        bp_saved = match_data[f'{player_prefix}_service_break_points_saved_pct']
        features['pressure_performance'] = 0.4 + bp_saved * 0.2
    else:
        features['pressure_performance'] = 0.50

    # Net game effectiveness
    if f'{player_prefix}_ta_net_won_pct' in match_data:
        features['net_effectiveness'] = match_data[f'{player_prefix}_ta_net_won_pct']
    elif f'{player_prefix}_net_points_won_pct' in match_data:
        features['net_effectiveness'] = match_data[f'{player_prefix}_net_points_won_pct']
    else:
        serve_eff = features.get('serve_effectiveness', 0.65)
        features['net_effectiveness'] = 0.55 + (serve_eff - 0.5) * 0.4

    return features

def extract_unified_match_context(match_data):
    """Extract match context with surface inference"""
    context = {}

    # Surface inference
    surface = match_data.get('surface', match_data.get('Surface', 'Unknown'))
    
    if surface == 'Unknown' or pd.isna(surface) or surface is None:
        tournament_round = str(match_data.get('tournament_round', '')).lower()
        tournament_name = str(match_data.get('Tournament', '')).lower()
        tournament_tier = str(match_data.get('tournament_tier', '')).lower()
        
        if 'wimbledon' in tournament_round or 'wimbledon' in tournament_name:
            surface = 'Grass'
        elif 'french' in tournament_round or 'roland garros' in tournament_round:
            surface = 'Clay'
        elif 'australian' in tournament_round or 'us open' in tournament_round:
            surface = 'Hard'
        elif 'masters' in tournament_tier or 'atp' in tournament_tier:
            surface = 'Hard'
        else:
            surface = 'Hard'
    
    context['surface'] = surface
    context['p1_ranking'] = match_data.get('p1_ranking', match_data.get('WRank'))
    context['p2_ranking'] = match_data.get('p2_ranking', match_data.get('LRank'))
    context['h2h_matches'] = match_data.get('h2h_matches', 0)
    context['p1_h2h_win_pct'] = match_data.get('p1_h2h_win_pct', 0.5)
    context['odds_p1'] = match_data.get('odds_p1', match_data.get('PSW'))
    context['odds_p2'] = match_data.get('odds_p2', match_data.get('PSL'))

    if context['odds_p1'] and context['odds_p2']:
        context['implied_prob_p1'] = 1 / context['odds_p1']
        context['implied_prob_p2'] = 1 / context['odds_p2']

    context['source_rank'] = match_data.get('source_rank', 3)
    
    # Data quality
    score = 0.5 if match_data.get('source_rank', 3) == 2 else 0.3
    api_stats = sum(1 for k in match_data.keys() if 'service_' in k and pd.notna(match_data.get(k, None)))
    if api_stats > 10:
        score += 0.2
    context['data_quality_score'] = min(score, 1.0)

    return context

class UnifiedBayesianTennisModel:
    """Simplified tennis prediction model"""
    
    def __init__(self, n_simulations=1000):
        self.n_simulations = n_simulations
        
        self.surface_adjustments = {
            "Clay": {"serve_advantage": 0.92, "rally_importance": 1.15},
            "Grass": {"serve_advantage": 1.15, "rally_importance": 0.85},
            "Hard": {"serve_advantage": 1.0, "rally_importance": 1.0}
        }

    def simulate_match(self, p1_features, p2_features, match_context, best_of=3):
        """Simple match simulation"""
        surface = match_context.get('surface', 'Hard')
        surface_adj = self.surface_adjustments.get(surface, self.surface_adjustments["Hard"])
        
        # Calculate player strengths
        p1_strength = (
            p1_features['serve_effectiveness'] * surface_adj["serve_advantage"] * 0.4 +
            (1 - p2_features['return_effectiveness']) * 0.3 +
            p1_features['pressure_performance'] * 0.3
        )
        
        p2_strength = (
            p2_features['serve_effectiveness'] * surface_adj["serve_advantage"] * 0.4 +
            (1 - p1_features['return_effectiveness']) * 0.3 +
            p2_features['pressure_performance'] * 0.3
        )
        
        # Ranking adjustment
        if match_context.get('p1_ranking') and match_context.get('p2_ranking'):
            rank_diff = match_context['p2_ranking'] - match_context['p1_ranking']
            if rank_diff > 0:  # p1 better ranked
                p1_strength += min(rank_diff * 0.002, 0.1)
        
        total = p1_strength + p2_strength
        return p1_strength / total if total > 0 else 0.5

def normalize_name(name):
    """Normalize player names"""
    if pd.isna(name):
        return ""
    return str(name).replace('.', '').lower().replace(' ', '_')

def build_composite_id(match_date, tourney_slug, p1_slug, p2_slug):
    """Build composite match ID"""
    if pd.isna(match_date):
        raise ValueError("match_date is NaT")
    ymd = pd.to_datetime(match_date).strftime("%Y%m%d")
    return f"{ymd}-{tourney_slug}-{p1_slug}-{p2_slug}"

def predict_match_unified(args, hist, jeff_data, defaults):
    """Simplified prediction function"""
    match_date = pd.to_datetime(args.date).date()
    
    # Generate name variations
    tournament_base = args.tournament.lower().strip()
    tournament_variations = [
        tournament_base,
        tournament_base.replace(' ', '_'),
        f"atp_{tournament_base}",
        f"atp {tournament_base}"
    ]
    
    def get_name_variations(player_name):
        base = normalize_name(player_name)
        variations = [base]
        parts = player_name.lower().split()
        if len(parts) >= 2:
            first = parts[0]
            last = parts[-1]
            variations.extend([
                f"{last}_{first[0]}",
                f"{first[0]}_{last}",
                f"{first}_{last}"
            ])
        return list(set(variations))
    
    p1_variations = get_name_variations(args.player1)
    p2_variations = get_name_variations(args.player2)
    
    # Try all combinations
    for tournament in tournament_variations:
        for p1 in p1_variations:
            for p2 in p2_variations:
                for player1, player2 in [(p1, p2), (p2, p1)]:
                    comp_id = f"{match_date.strftime('%Y%m%d')}-{tournament}-{player1}-{player2}"
                    
                    row = hist[hist["composite_id"] == comp_id]
                    
                    if not row.empty:
                        match_row = row.iloc[0]
                        match_dict = match_row.to_dict()
                        
                        # If players swapped, correct features
                        if (player1, player2) == (p2, p1):
                            swapped_dict = {}
                            for key, value in match_dict.items():
                                if key.startswith('winner_'):
                                    swapped_dict[key.replace('winner_', 'loser_')] = value
                                elif key.startswith('loser_'):
                                    swapped_dict[key.replace('loser_', 'winner_')] = value
                                else:
                                    swapped_dict[key] = value
                            match_dict = swapped_dict
                        
                        # Extract features and predict
                        p1_features = extract_unified_features(match_dict, 'winner')
                        p2_features = extract_unified_features(match_dict, 'loser')
                        match_context = extract_unified_match_context(match_dict)
                        
                        # Run prediction
                        model = UnifiedBayesianTennisModel()
                        prob = model.simulate_match(p1_features, p2_features, match_context, best_of=args.best_of)
                        
                        return prob
    
    return None
