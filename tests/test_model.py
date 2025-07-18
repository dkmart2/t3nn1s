import pandas as pd
import numpy as np

def generate_synthetic_point_data(n_matches=50, points_per_match=100):
    """Generate realistic synthetic point data"""
    data = []
    surfaces = ['Hard', 'Clay', 'Grass']

    for match_id in range(n_matches):
        surface = np.random.choice(surfaces)
        for point_num in range(points_per_match):
            server = np.random.choice([1, 2])
            winner = server if np.random.random() < 0.65 else (3 - server)

            data.append({
                'match_id': f'match_{match_id}',
                'Pt': point_num + 1,
                'Svr': server,
                'PtWinner': winner,
                'surface': surface,
                'is_break_point': np.random.random() < 0.08,
                'is_set_point': np.random.random() < 0.05,
                'is_match_point': np.random.random() < 0.02,
                'rallyCount': np.random.poisson(4) + 1,
                'p1_games': np.random.randint(0, 7),
                'p2_games': np.random.randint(0, 7),
                'p1_sets': np.random.randint(0, 3),
                'p2_sets': np.random.randint(0, 3),
            })

    df = pd.DataFrame(data)
    print(f"DEBUG: Generated DataFrame has columns: {list(df.columns)}")
    return df