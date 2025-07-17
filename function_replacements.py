
# Clean function replacements for tennis_updated.py
# Copy these functions to replace the existing ones

def extract_unified_features(match_data, player_prefix):
    """Extract features with fallbacks across all data sources - FIXED VERSION"""
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

    # Return effectiveness (API > Tennis Abstract > Jeff > defaults)  
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

    # Winners rate - estimate from serve/return effectiveness for API data
    if f'{player_prefix}_ta_overview_winners' in match_data:
        serve_pts = match_data.get(f'{player_prefix}_ta_serve_points', 80)
        return_pts = match_data.get(f'{player_prefix}_ta_return_points', 70)
        total_pts = serve_pts + return_pts
        features['winners_rate'] = match_data[f'{player_prefix}_ta_overview_winners'] / total_pts if total_pts > 0 else 0.20
    elif f'{player_prefix}_winners_total' in match_data:
        serve_pts = match_data.get(f'{player_prefix}_serve_pts', 80)
        features['winners_rate'] = match_data[f'{player_prefix}_winners_total'] / serve_pts if serve_pts > 0 else 0.20
    else:
        # Estimate from service effectiveness for API data
        serve_eff = features.get('serve_effectiveness', 0.65)
        features['winners_rate'] = 0.1 + (serve_eff - 0.5) * 0.4

    # Unforced errors rate - estimate from double faults and second serve performance
    if f'{player_prefix}_ta_overview_unforced' in match_data:
        serve_pts = match_data.get(f'{player_prefix}_ta_serve_points', 80)
        return_pts = match_data.get(f'{player_prefix}_ta_return_points', 70) 
        total_pts = serve_pts + return_pts
        features['unforced_rate'] = match_data[f'{player_prefix}_ta_overview_unforced'] / total_pts if total_pts > 0 else 0.18
    elif f'{player_prefix}_unforced_errors' in match_data:
        serve_pts = match_data.get(f'{player_prefix}_serve_pts', 80)
        features['unforced_rate'] = match_data[f'{player_prefix}_unforced_errors'] / serve_pts if serve_pts > 0 else 0.18
    else:
        # Estimate from double faults and serve performance
        dfs = match_data.get(f'{player_prefix}_service_double_faults', 3)
        serve_total = match_data.get(f'{player_prefix}_points_service_points_won_total', 74)
        df_rate = dfs / serve_total if serve_total > 0 else 0.04
        features['unforced_rate'] = 0.12 + df_rate * 2

    # Pressure performance from break points 
    if f'{player_prefix}_pressure_performance' in match_data:
        features['pressure_performance'] = match_data[f'{player_prefix}_pressure_performance']
    elif f'{player_prefix}_key_points_serve_won_pct' in match_data:
        features['pressure_performance'] = match_data[f'{player_prefix}_key_points_serve_won_pct']
    elif f'{player_prefix}_service_break_points_saved_pct' in match_data:
        bp_saved = match_data[f'{player_prefix}_service_break_points_saved_pct']
        features['pressure_performance'] = 0.4 + bp_saved * 0.2
    else:
        features['pressure_performance'] = 0.50

    # Net game effectiveness - estimate from serve effectiveness for API data
    if f'{player_prefix}_ta_net_won_pct' in match_data:
        features['net_effectiveness'] = match_data[f'{player_prefix}_ta_net_won_pct']
    elif f'{player_prefix}_net_points_won_pct' in match_data:
        features['net_effectiveness'] = match_data[f'{player_prefix}_net_points_won_pct']
    else:
        serve_eff = features.get('serve_effectiveness', 0.65)
        features['net_effectiveness'] = 0.55 + (serve_eff - 0.5) * 0.4

    return features


def extract_unified_match_context(match_data):
    """Extract match context from any data source - FIXED VERSION"""
    context = {}

    # Surface inference from tournament
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

    # Rankings
    context['p1_ranking'] = match_data.get('p1_ranking', match_data.get('WRank'))
    context['p2_ranking'] = match_data.get('p2_ranking', match_data.get('LRank'))

    # H2H data
    context['h2h_matches'] = match_data.get('h2h_matches', 0)
    context['p1_h2h_win_pct'] = match_data.get('p1_h2h_win_pct', 0.5)

    # Odds
    context['odds_p1'] = match_data.get('odds_p1', match_data.get('PSW'))
    context['odds_p2'] = match_data.get('odds_p2', match_data.get('PSL'))

    if context['odds_p1'] and context['odds_p2']:
        context['implied_prob_p1'] = 1 / context['odds_p1']
        context['implied_prob_p2'] = 1 / context['odds_p2']

    # Data quality
    context['source_rank'] = match_data.get('source_rank', 3)
    context['data_quality_score'] = calculate_data_quality_unified(match_data)

    return context


def calculate_data_quality_unified(match_data):
    """Calculate data quality score based on available features - FIXED VERSION"""
    score = 0.3  # Base score

    # Source quality bonus
    source_rank = match_data.get('source_rank', 3)
    if source_rank == 1:  # Tennis Abstract
        score += 0.4
    elif source_rank == 2:  # API-Tennis
        score += 0.35  # Increased for API data
    elif source_rank == 3:  # Jeff/Tennis files
        score += 0.1

    # API statistics availability bonus
    api_service_cols = sum(1 for k in match_data.keys() if 'service_' in k and pd.notna(match_data.get(k)) and match_data.get(k) != 0)
    api_return_cols = sum(1 for k in match_data.keys() if 'return_' in k and pd.notna(match_data.get(k)) and match_data.get(k) != 0)
    
    if api_service_cols > 5 and api_return_cols > 5:
        score += 0.15  # Comprehensive API statistics
    elif api_service_cols > 3 or api_return_cols > 3:
        score += 0.1   # Partial API statistics

    # Feature availability bonus
    ta_features = sum(1 for k in match_data.keys() if 'ta_' in k)
    jeff_features = sum(1 for k in match_data.keys() if any(x in k for x in ['serve_pts', 'winners_total', 'pressure_performance']))

    if ta_features > 10:
        score += 0.1
    elif jeff_features > 8:
        score += 0.05

    return min(score, 1.0)
