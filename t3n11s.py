import os
from unidecode import unidecode
<truncated__content/>data) > 0:
        cutoff_data = historical_data[historical_data["date"] >= start_date]
        if len(cutoff_data) > 0:
            for key in cutoff_data["event_key"].dropna():
                converted_key = safe_int_convert(key)
                if converted_key is not None:
                    existing_keys.add(converted_key)

    # initialise Tennis‑Abstract scraper once per run
    scraper = TennisAbstractScraper()
    api_matches = []
    date_range = list(pd.date_range(start_date, date.today()))

    for d in date_range:
        day = d.date()
        try:
            fixtures = get_fixtures_for_date(day)
            day_urls = charting_urls_for_day(day)
            # DEBUG ─ how many TA charting pages exist for the day
            print(f"  {day}: {len(day_urls)} charting URLs")
        except Exception as exc:
            print(f"    charting-index fetch failed {day}: {exc}")
            fixtures = []
            day_urls = []
        if fixtures:
            print(f"  {day}: {len(fixtures)} finished matches")

            for fixture in fixtures:
                try:
                    event_key = safe_int_convert(fixture.get("event_key"))
                    if event_key is None or event_key in existing_keys:
                        continue

                    p1_name = fixture["event_first_player"]
                    p2_name = fixture["event_second_player"]
                    winner = p1_name if fixture.get("event_winner", "").startswith("First") else p2_name
                    loser = p2_name if winner == p1_name else p1_name

                    win_c = normalize_name(winner)
                    los_c = normalize_name(loser)
                    tour_c = normalize_tournament_name(fixture.get("tournament_name", "Unknown"))

                    match_record = {
                        "event_key": event_key,
                        "Date": pd.to_datetime(fixture["event_date"]),
                        "date": pd.to_datetime(fixture["event_date"]).date(),
                        "Tournament": fixture.get("tournament_name", "Unknown"),
                        "round": fixture.get("tournament_round", ""),
                        "Surface": fixture.get("court_surface", "Hard"),
                        "Winner": winner,
                        "Loser": loser,
                        "source_rank": 1,
                        "gender": "W" if "wta" in fixture.get("event_type_type", "").lower() else "M",
                        "winner_canonical": win_c,
                        "loser_canonical": los_c,
                        "tournament_canonical": tour_c,
                        "ta_stats_json": pd.NA,  # placeholder, updated later when TA scrape succeeds
                    }

                    match_record["composite_id"] = build_composite_id(
                        match_record["date"], tour_c, win_c, los_c
                    )

                    # ─── Tennis‑Abstract scraping (11 sections) ───────────────
                    ta_url = None
                    ta_stats = {}
                    # ─── robust Tennis‑Abstract URL match ──────────────────
                    def _canon_tag(name: str) -> str:
                        """Approximate canonical tag used in TA filenames."""
                        return unidecode(
                            name.lower()
                                .replace('-', '_')
                                .replace(' ', '_')
                                .replace("'", "")
                        )

                    win_tag   = _canon_tag(winner)
                    los_tag   = _canon_tag(loser)
                    win_lname = win_tag.split('_')[0]
                    los_lname = los_tag.split('_')[0]

                    # Attempt strict tag‑pair match first, then fall back to last‑name pair
                    ta_url = None
                    for url in day_urls:
                        fname = os.path.basename(url).lower()

                        # strict: full canonical tags both present (any order)
                        cond_strict = (
                            (win_tag in fname and los_tag in fname) or
                            (los_tag in fname and win_tag in fname)
                        )

                        # loose: both last names present (avoid same‑name clash)
                        cond_loose = (
                            (win_lname != los_lname) and
                            (
                                (win_lname in fname and los_lname in fname) or
                                (los_lname in fname and win_lname in fname)
                            )
                        )

                        if cond_strict or cond_loose:
                            ta_url = url
                            break

                    # DEBUG ─ report matching result
                    if ta_url:
                        print(f"      TA url matched: {os.path.basename(ta_url)}")
                    else:
                        print(f"      No TA url for {winner} vs {loser}")

                    if ta_url:
                        try:
                            ta_stats = {
                                "stats_overview":           scraper.scrape_stats_overview(ta_url),
                                "serve_influence":          scraper.scrape_serve_influence(ta_url),
                                "serve_breakdown":          scraper.scrape_serve_breakdown(ta_url),
                                "return_breakdown":         scraper.scrape_return_breakdown(ta_url),
                                "key_points":               scraper.scrape_key_point_outcomes(ta_url),
                                "rally_outcomes":           scraper.scrape_rally_outcomes(ta_url),
                                "net_points":               scraper.scrape_net_points(ta_url),
                                "shot_types":               scraper.scrape_shot_types(ta_url),
                                "shot_direction":           scraper.scrape_shot_direction(ta_url),
                                "serve_stats":              scraper.scrape_serve_statistics_overview(ta_url),
                                "pointlog":                 scraper.scrape_pointlog(ta_url),
                            }
                        except Exception as exc:
                            print(f"      TA scrape failed {ta_url}: {exc}")
                        # persist raw TA data (JSON) for downstream parsing, but only if ta_stats is not empty
                        if ta_stats:
                            match_record["ta_url"]        = ta_url
                            match_record["ta_stats_json"] = json.dumps(ta_stats)

                    embedded_stats = extract_embedded_statistics(fixture)
                    match_record.update(embedded_stats)

                    odds1, odds2 = get_match_odds(event_key, day)
                    match_record["api_odds_home"] = odds1
                    match_record["api_odds_away"] = odds2

                    league = "WTA" if match_record["gender"] == "W" else "ATP"
                    rankings = get_player_rankings(day, league)
                    p1_key = int(fixture.get("first_player_key", 0))
                    p2_key = int(fixture.get("second_player_key", 0))

                    if winner == p1_name:
                        match_record["WRank"] = rankings.get(p1_key, pd.NA)
                        match_record["LRank"] = rankings.get(p2_key, pd.NA)
                    else:
                        match_record["WRank"] = rankings.get(p2_key, pd.NA)
                        match_record["LRank"] = rankings.get(p1_key, pd.NA)

                    api_matches.append(match_record)
                    existing_keys.add(event_key)

                    time.sleep(0.1)

                except Exception as e:
                    print(f"    Error processing match {fixture.get('event_key', 'unknown')}: {e}")
                    continue

        except Exception as e:
            print(f"  Error processing day {day}: {e}")
            continue

    if api_matches:
        try:
            api_df = pd.DataFrame(api_matches)

            for col in historical_data.columns:
                if col not in api_df.columns:
                    api_df[col] = pd.NA

            for col in api_df.columns:
                if col not in historical_data.columns:
                    historical_data[col] = pd.NA

            api_df = api_df.reindex(columns=historical_data.columns, fill_value=pd.NA)
            combined_data = pd.concat([historical_data, api_df], ignore_index=True)

            if "source_rank" not in combined_data.columns:
                combined_data["source_rank"] = 2
            combined_data["source_rank"] = combined_data["source_rank"].fillna(2)

            dedup_keys = ["event_key", "composite_id"]
            final_data = (
                combined_data
                .sort_values("source_rank")
                .drop_duplicates(subset=dedup_keys, keep="first")
                .reset_index(drop=True)
            )

            print(f"✓ Successfully integrated {len(api_df)} API matches")
            return final_data

        except Exception as e:
            print(f"Error merging API data: {e}")
            return historical_data
    else:
        print("No new API data to merge")
        return historical_data


# 2.7 Data Generation and Cache Management
def generate_comprehensive_historical_data(fast=True, n_sample=500):
    """Generate comprehensive historical data with API integration"""
    print("=== STARTING DATA GENERATION ===")

    print("Step 1: Loading Jeff's comprehensive data...")
    try:
        jeff_data = load_jeff_comprehensive_data()
        if not jeff_data or ('men' not in jeff_data and 'women' not in jeff_data):
            print("ERROR: Jeff data loading failed")
            return pd.DataFrame(), {}, {}
        print(f"✓ Jeff data loaded successfully")
    except Exception as e:
        print(f"ERROR loading Jeff data: {e}")
        return pd.DataFrame(), {}, {}

    print("Step 2: Calculating weighted defaults...")
    try:
        weighted_defaults = calculate_comprehensive_weighted_defaults(jeff_data)
        if not weighted_defaults:
            print("ERROR: Weighted defaults calculation failed")
            return pd.DataFrame(), jeff_data, {}
        print(f"✓ Weighted defaults calculated")
    except Exception as e:
        print(f"ERROR calculating weighted defaults: {e}")
        return pd.DataFrame(), jeff_data, {}

    print("Step 3: Loading tennis match data...")
    try:
        tennis_data = load_all_tennis_data()
        if tennis_data.empty:
            print("ERROR: No tennis data loaded")
            return pd.DataFrame(), jeff_data, weighted_defaults
        print(f"✓ Tennis data loaded: {len(tennis_data)} matches")

        if fast:
            total_rows = len(tennis_data)
            take = min(n_sample, total_rows)
            tennis_data = tennis_data.sample(take, random_state=1).reset_index(drop=True)
            print(f"[FAST MODE] Using sample of {take}/{total_rows} rows")
    except Exception as e:
        print(f"ERROR loading tennis data: {e}")
        return pd.DataFrame(), jeff_data, weighted_defaults

    print("Step 4: Processing tennis data...")
    try:
        tennis_data['winner_canonical'] = tennis_data['Winner'].apply(normalize_name)
        tennis_data['loser_canonical'] = tennis_data['Loser'].apply(normalize_name)
        tennis_data['tournament_canonical'] = tennis_data['Tournament'].apply(normalize_tournament_name)
        tennis_data['Date'] = pd.to_datetime(tennis_data['Date'], errors='coerce')
        tennis_data['date'] = tennis_data['Date'].dt.date

        tennis_data['composite_id'] = tennis_data.apply(
            lambda r: build_composite_id(
                r['date'], r['tournament_canonical'], r['winner_canonical'], r['loser_canonical']
            ), axis=1
        )

        tennis_data['tennis_data_odds1'] = pd.to_numeric(tennis_data.get('PSW', 0), errors='coerce')
        tennis_data['tennis_data_odds2'] = pd.to_numeric(tennis_data.get('PSL', 0), errors='coerce')

        if 'WRank' in tennis_data.columns and 'LRank' in tennis_data.columns:
            tennis_data['rank_difference'] = abs(pd.to_numeric(tennis_data['WRank'], errors='coerce') -
                                                 pd.to_numeric(tennis_data['LRank'], errors='coerce'))

        print(f"✓ Tennis data processed")
    except Exception as e:
        print(f"ERROR processing tennis data: {e}")
        return pd.DataFrame(), jeff_data, weighted_defaults

    print("Step 5: Adding Jeff feature columns...")
    try:
        men_feats = set(weighted_defaults.get('men', {}).keys())
        women_feats = set(weighted_defaults.get('women', {}).keys())
        all_jeff_features = sorted(men_feats.union(women_feats))

        if not all_jeff_features:
            raise ValueError("No features available in weighted_defaults")

        missing_cols_dict = {}
        for feat in all_jeff_features:
            w_col = f"winner_{feat}"
            l_col = f"loser_{feat}"

            if w_col not in tennis_data.columns:
                missing_cols_dict[w_col] = np.full(len(tennis_data), np.nan, dtype="float64")
            if l_col not in tennis_data.columns:
                missing_cols_dict[l_col] = np.full(len(tennis_data), np.nan, dtype="float64")

        if missing_cols_dict:
            tennis_data = pd.concat(
                [tennis_data, pd.DataFrame(missing_cols_dict, index=tennis_data.index)],
                axis=1
            )

        print(f"✓ Added/verified {len(all_jeff_features) * 2} feature columns")
    except Exception as e:
        print(f"ERROR adding feature columns: {e}")
        return pd.DataFrame(), jeff_data, weighted_defaults

    print("Step 6: Extracting Jeff features...")
    try:
        total_matches = len(tennis_data)
        matches_with_jeff_features = 0

        for idx, row in tennis_data.iterrows():
            if idx % 100 == 0:
                print(f"  Processing match {idx}/{total_matches}")

            try:
                gender = row['gender']

                if row['date'] <= date(2025, 6, 10):
                    winner_features = extract_comprehensive_jeff_features(
                        row['winner_canonical'], gender, jeff_data, weighted_defaults
                    )
                    loser_features = extract_comprehensive_jeff_features(
                        row['loser_canonical'], gender, jeff_data, weighted_defaults
                    )

                    for feature_name, feature_value in winner_features.items():
                        col_name = f'winner_{feature_name}'
                        if col_name in tennis_data.columns:
                            tennis_data.at[idx, col_name] = feature_value

                    for feature_name, feature_value in loser_features.items():
                        col_name = f'loser_{feature_name}'
                        if col_name in tennis_data.columns:
                            tennis_data.at[idx, col_name] = feature_value

                    if winner_features and loser_features:
                        matches_with_jeff_features += 1

            except Exception as e:
                if idx < 5:
                    print(f"  Warning: Error processing match {idx}: {e}")
                continue

        print(f"✓ Jeff features extracted for {matches_with_jeff_features}/{total_matches} matches")
    except Exception as e:
        print(f"ERROR extracting Jeff features: {e}")
        return pd.DataFrame(), jeff_data, weighted_defaults

    print(f"=== DATA GENERATION COMPLETE ===")
    print(f"Final data shape: {tennis_data.shape}")

    return tennis_data, jeff_data, weighted_defaults


def save_to_cache(historical_data, jeff_data, weighted_defaults):
    """Save data to cache"""
    print("\n=== SAVING TO CACHE ===")
    numeric_cols = ["MaxW", "MaxL", "AvgW", "AvgL", "PSW", "PSL"]
    for col in numeric_cols:
        if col in historical_data.columns:
            historical_data[col] = pd.to_numeric(historical_data[col], errors="coerce")
    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        historical_data.to_parquet(HD_PATH, index=False)
        print("✓ Historical data saved")

        with open(JEFF_PATH, "wb") as f:
            pickle.dump(jeff_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✓ Jeff data saved")

        with open(DEF_PATH, "wb") as f:
            pickle.dump(weighted_defaults, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✓ Weighted defaults saved")

        return True
    except Exception as e:
        print(f"ERROR saving cache: {e}")
        return False


def load_from_cache():
    """Load data from cache if available"""
    if (os.path.exists(HD_PATH) and
            os.path.exists(JEFF_PATH) and
            os.path.exists(DEF_PATH)):

        print("Loading from cache...")
        historical_data = pd.read_parquet(HD_PATH)

        with open(JEFF_PATH, "rb") as f:
            jeff_data = pickle.load(f)

        with open(DEF_PATH, "rb") as f:
            weighted_defaults = pickle.load(f)

        return historical_data, jeff_data, weighted_defaults
    else:
        return None, None, None

#%
# ============================================================================
# 3. SIMULATION ENGINE
# ============================================================================

# 3.1 Base Bayesian Tennis Model
class BayesianTennisModel:
    def __init__(self):
        self.simulation_count = 10000
        self.jeff_data = None
        self.historical_data = None

    def default_priors(self):
        return {
            'elo_mean': 1500,
            'elo_std': 200,
            'hold_prob': 0.65,
            'break_prob': 0.35,
            'surface': 'Hard',
            'form_factor': 1.0,
            'confidence': 0.1
        }

    def extract_refined_priors(self, player_canonical, gender, surface, reference_date):
        player_matches = self.historical_data[
            (self.historical_data['winner_canonical'] == player_canonical) |
            (self.historical_data['loser_canonical'] == player_canonical)
            ].copy()

        if len(player_matches) == 0:
            return self.default_priors()

        surface_matches = player_matches[player_matches['Surface'] == surface]
        if len(surface_matches) < 5:
            surface_matches = player_matches

        recent_matches = surface_matches.tail(20).copy()
        recent_matches['days_ago'] = (pd.to_datetime(reference_date) - pd.to_datetime(recent_matches['Date'])).dt.days
        weights = np.exp(-0.05 * recent_matches['days_ago'])

        base_elo = self.get_player_weighted_elo(player_canonical, surface, reference_date)
        surface_factor = self.calculate_surface_adaptation(player_canonical, surface)
        elo_prior = base_elo * surface_factor

        jeff_features = extract_jeff_features(player_canonical, gender, self.jeff_data)

        serve_pts = jeff_features['serve_pts']
        serve_won = jeff_features['first_won'] + jeff_features['second_won']
        hold_prob = serve_won / serve_pts if serve_pts > 0 else 0.65

        return_pts = jeff_features['return_pts_won']
        total_return_pts = serve_pts
        break_prob = (1 - return_pts / total_return_pts) if total_return_pts > 0 else 0.35

        return {
            'elo_mean': elo_prior,
            'elo_std': 150,
            'hold_prob': min(0.95, max(0.3, hold_prob)),
            'break_prob': max(0.05, min(0.7, break_prob)),
            'surface': surface,
            'form_factor': self.calculate_form_spike(recent_matches, weights, player_canonical),
            'confidence': max(0.05, min(1.0, len(recent_matches) / 15))
        }

    def get_player_ranking(self, player_canonical, date):
        """Get player ranking at specific date"""
        date_obj = pd.to_datetime(date)

        player_matches = self.historical_data[
            ((self.historical_data['winner_canonical'] == player_canonical) |
             (self.historical_data['loser_canonical'] == player_canonical)) &
            (pd.to_datetime(self.historical_data['Date']) <= date_obj)
            ].sort_values('Date', ascending=False)

        if len(player_matches) == 0:
            return 999

        latest_match = player_matches.iloc[0]

        if latest_match['winner_canonical'] == player_canonical:
            return latest_match.get('WRank', 999)
        else:
            return latest_match.get('LRank', 999)

    def calculate_form_spike(self, recent_matches, weights, player_canonical):
        if len(recent_matches) == 0:
            return 1.0

        wins = (recent_matches['winner_canonical'] == player_canonical).astype(int)
        weighted_win_rate = np.average(wins, weights=weights)

        avg_opponent_rank = recent_matches['LRank'].fillna(recent_matches['WRank']).mean()
        player_rank = recent_matches['WRank'].fillna(recent_matches['LRank']).iloc[-1]

        if pd.notna(avg_opponent_rank) and pd.notna(player_rank):
            rank_diff = player_rank - avg_opponent_rank
            expected_win_rate = 1 / (1 + 10 ** (rank_diff / 400))
            form_spike = min(2.0, weighted_win_rate / max(0.1, expected_win_rate))
        else:
            form_spike = 1.0

        return form_spike

    def get_player_weighted_elo(self, player_canonical, surface, reference_date):
        recent_match = self.historical_data[
            ((self.historical_data['winner_canonical'] == player_canonical) |
             (self.historical_data['loser_canonical'] == player_canonical)) &
            (self.historical_data['Surface'] == surface)
            ].tail(1)

        if len(recent_match) > 0 and 'BlendScore' in recent_match.columns:
            blend_score = recent_match['BlendScore'].iloc[0]
            return 1500 + blend_score * 50

        any_surface_match = self.historical_data[
            (self.historical_data['winner_canonical'] == player_canonical) |
            (self.historical_data['loser_canonical'] == player_canonical)
            ].tail(1)

        if len(any_surface_match) > 0 and 'BlendScore' in any_surface_match.columns:
            return 1500 + any_surface_match['BlendScore'].iloc[0] * 200

        return 1500

    def calculate_surface_adaptation(self, player_canonical, target_surface):
        player_matches = self.historical_data[
            (self.historical_data['winner_canonical'] == player_canonical) |
            (self.historical_data['loser_canonical'] == player_canonical)
            ].copy()

        if len(player_matches) < 10:
            return 1.0

        surface_matches = player_matches[player_matches['Surface'] == target_surface]
        if len(surface_matches) < 3:
            return 1.0

        surface_wins = (surface_matches['winner_canonical'] == player_canonical).sum()
        surface_win_rate = surface_wins / len(surface_matches)

        total_wins = (player_matches['winner_canonical'] == player_canonical).sum()
        baseline_win_rate = total_wins / len(player_matches)

        if baseline_win_rate == 0:
            return 1.0

        adaptation_ratio = surface_win_rate / baseline_win_rate
        return max(0.7, min(1.5, adaptation_ratio))

    # 3.2 Layer 2: Contextual Adjustments
    def apply_contextual_adjustments(self, priors, player_canonical, opponent_canonical, match_context):
        """Layer 2: Contextual Bayesian adjustments for fatigue, injury, motivation"""
        adjusted_priors = priors.copy()

        fatigue_penalty = self.calculate_fatigue_index(player_canonical, match_context['reference_date'])
        adjusted_priors['hold_prob'] *= (1 - fatigue_penalty * 0.15)
        adjusted_priors['elo_std'] *= (1 + fatigue_penalty * 0.3)

        injury_factor = self.get_injury_factor(player_canonical, match_context['reference_date'])
        adjusted_priors['hold_prob'] *= injury_factor
        adjusted_priors['break_prob'] *= (2 - injury_factor)

        form_sustainability = self.calculate_form_sustainability(player_canonical, match_context)
        if adjusted_priors['form_factor'] > 1.2:
            sustainability_discount = 1 - ((adjusted_priors['form_factor'] - 1) * (1 - form_sustainability))
            adjusted_priors['hold_prob'] *= sustainability_discount
            adjusted_priors['elo_mean'] *= sustainability_discount

        opponent_elo = self.estimate_opponent_elo(opponent_canonical, match_context)
        elo_diff = adjusted_priors['elo_mean'] - opponent_elo
        quality_adjustment = 1 / (1 + np.exp(-elo_diff / 200))
        adjusted_priors['break_prob'] *= quality_adjustment

        return adjusted_priors

    def calculate_fatigue_index(self, player_canonical, reference_date):
        """Fatigue based on recent match load and recovery time"""
        recent_matches = self.get_recent_matches(player_canonical, reference_date, days=14)

        if len(recent_matches) == 0:
            return 0.0

        fatigue_score = 0
        for _, match in recent_matches.iterrows():
            days_ago = (pd.to_datetime(reference_date) - pd.to_datetime(match['Date'])).days
            match_duration = match.get('minutes', 120)

            fatigue_contribution = (match_duration / 60) * np.exp(-0.1 * days_ago)
            fatigue_score += fatigue_contribution

        return min(1.0, fatigue_score / 10)

    def get_injury_factor(self, player_canonical, reference_date):
        """Player-specific injury fragility scoring"""
        injury_prone_players = {
            'nadal_r': 0.85,
            'murray_a': 0.80,
            'thiem_d': 0.75,
            'badosa_p': 0.70
        }

        base_factor = injury_prone_players.get(player_canonical, 0.95)
        recent_retirements = self.check_recent_retirements(player_canonical, reference_date)
        if recent_retirements > 0:
            base_factor *= (0.8 ** recent_retirements)

        return max(0.5, base_factor)

    def calculate_form_sustainability(self, player_canonical, match_context):
        """Form spike sustainability based on opponent quality and win quality"""
        recent_matches = self.get_recent_matches(player_canonical, match_context['reference_date'], days=21)

        if len(recent_matches) < 3:
            return 0.5

        quality_scores = []
        for _, match in recent_matches.iterrows():
            opponent_rank = match['LRank'] if match['winner_canonical'] == player_canonical else match['WRank']
            win_quality = 1 / (1 + opponent_rank / 100) if pd.notna(opponent_rank) else 0.5
            quality_scores.append(win_quality)

        avg_opponent_quality = np.mean(quality_scores)
        consistency = 1 - np.std(quality_scores)

        return min(1.0, avg_opponent_quality * consistency)

    def estimate_opponent_elo(self, opponent_canonical, match_context):
        """Quick opponent Elo estimation for quality weighting"""
        opponent_priors = self.extract_refined_priors(
            opponent_canonical,
            match_context['gender'],
            match_context['surface'],
            match_context['reference_date']
        )
        return opponent_priors['elo_mean']

    def get_recent_matches(self, player_canonical, reference_date, days=14):
        try:
            cutoff_date = pd.to_datetime(reference_date) - pd.Timedelta(days=days)

            player_matches = self.historical_data[
                ((self.historical_data['winner_canonical'] == player_canonical) |
                 (self.historical_data['loser_canonical'] == player_canonical))
            ].copy()

            if len(player_matches) == 0:
                return player_matches

            player_matches['Date'] = pd.to_datetime(player_matches['Date'].astype(str), errors='coerce')
            player_matches = player_matches.dropna(subset=['Date'])
            player_matches = player_matches[player_matches['Date'] >= cutoff_date]

            return player_matches.sort_values('Date')
        except:
            return pd.DataFrame()

    def check_recent_retirements(self, player_canonical, reference_date):
        """Count recent retirements/walkovers"""
        return 0

    # 3.3 Layer 3: Monte Carlo Simulation
    def simulate_match(self, player1_priors, player2_priors, best_of=3, tiebreak_sets=[1, 2, 3]):
        """Layer 3: Monte Carlo match simulation with Bayesian priors"""
        wins = 0
        simulations = self.simulation_count

        for _ in range(simulations):
            sets_won = [0, 0]

            while max(sets_won) < (best_of + 1) // 2:
                set_winner = self.simulate_set(
                    player1_priors,
                    player2_priors,
                    tiebreak=len([s for s in sets_won if s > 0]) + 1 in tiebreak_sets
                )
                sets_won[set_winner] += 1

            if sets_won[0] > sets_won[1]:
                wins += 1

        return wins / simulations

    def simulate_set(self, p1_priors, p2_priors, tiebreak=True):
        """Simulate single set with service alternation"""
        games = [0, 0]
        server = 0

        while True:
            if server == 0:
                hold_prob = p1_priors['hold_prob']
                game_winner = 0 if np.random.random() < hold_prob else 1
            else:
                hold_prob = p2_priors['hold_prob']
                game_winner = 1 if np.random.random() < hold_prob else 0

            games[game_winner] += 1
            server = 1 - server

            if games[0] >= 6 and games[0] - games[1] >= 2:
                return 0
            elif games[1] >= 6 and games[1] - games[0] >= 2:
                return 1
            elif games[0] == 6 and games[1] == 6 and tiebreak:
                return self.simulate_tiebreak(p1_priors, p2_priors)

    def simulate_tiebreak(self, p1_priors, p2_priors):
        """Simulate tiebreak with point-by-point serve alternation"""
        points = [0, 0]
        server = 0
        serve_count = 0

        while True:
            if server == 0:
                hold_prob = p1_priors['hold_prob']
                point_winner = 0 if np.random.random() < hold_prob else 1
            else:
                hold_prob = p2_priors['hold_prob']
                point_winner = 1 if np.random.random() < hold_prob else 0

            points[point_winner] += 1
            serve_count += 1

            if serve_count == 1 or serve_count % 2 == 0:
                server = 1 - server

            if points[0] >= 7 and points[0] - points[1] >= 2:
                return 0
            elif points[1] >= 7 and points[1] - points[0] >= 2:
                return 1

    def evaluate_predictions(self, test_data):
        """Evaluate model accuracy on test dataset"""
        correct = 0
        total = 0

        for _, match in test_data.iterrows():
            prob = self.predict_match_outcome(
                match['winner_canonical'],
                match['loser_canonical'],
                match['Surface'],
                match['gender'],
                match['Date']
            )

            predicted_winner = match['winner_canonical'] if prob > 0.5 else match['loser_canonical']
            actual_winner = match['winner_canonical']

            if predicted_winner == actual_winner:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0

    def predict_match_outcome(self, player1_canonical, player2_canonical, surface, gender, date):
        p1_priors = self.extract_refined_priors(player1_canonical, gender, surface, date)
        p2_priors = self.extract_refined_priors(player2_canonical, gender, surface, date)

        base_prob = self.simulate_match(p1_priors, p2_priors)

        p1_rank = self.get_player_ranking(player1_canonical, date)
        p2_rank = self.get_player_ranking(player2_canonical, date)
        ranking_prob = self.calculate_ranking_differential_odds(p1_rank, p2_rank)

        ranking_diff = p1_rank - p2_rank
        upset_adjustment = self.calculate_upset_frequency(ranking_diff, surface, self.historical_data)

        surface_ratio = self.calculate_surface_performance_ratio(player1_canonical, surface, player2_canonical, date)

        calibrated_prob = (0.6 * base_prob + 0.25 * ranking_prob + 0.15 * surface_ratio) * (1 - upset_adjustment * 0.1)

        return max(0.05, min(0.95, calibrated_prob))

    def calculate_ranking_differential_odds(self, p1_ranking, p2_ranking):
        """Convert ranking differential to implied probability"""
        if p1_ranking == 0 or p2_ranking == 0:
            return 0.5

        ranking_diff = p2_ranking - p1_ranking

        if ranking_diff > 50:
            return 0.85
        elif ranking_diff > 20:
            return 0.75
        elif ranking_diff > 10:
            return 0.65
        elif ranking_diff > 0:
            return 0.55
        elif ranking_diff > -10:
            return 0.45
        elif ranking_diff > -20:
            return 0.35
        elif ranking_diff > -50:
            return 0.25
        else:
            return 0.15

    def calculate_upset_frequency(self, ranking_diff, surface, historical_data):
        """Calculate upset frequency by ranking differential and surface"""
        upset_matches = historical_data[
            ((historical_data['WRank'] - historical_data['LRank']) > ranking_diff) &
            (historical_data['Surface'] == surface)
            ]

        total_matches = historical_data[
            (abs(historical_data['WRank'] - historical_data['LRank']) >= abs(ranking_diff)) &
            (historical_data['Surface'] == surface)
            ]

        if len(total_matches) < 10 and surface != 'fallback':
            return self.calculate_upset_frequency(ranking_diff, 'fallback', historical_data)

        if surface == 'fallback':
            upset_matches = historical_data[
                (historical_data['WRank'] - historical_data['LRank']) > ranking_diff
                ]
            total_matches = historical_data[
                abs(historical_data['WRank'] - historical_data['LRank']) >= abs(ranking_diff)
                ]

        if len(total_matches) == 0:
            return 0.1

        upset_rate = len(upset_matches) / len(total_matches)
        return min(0.45, max(0.05, upset_rate))

    def calculate_surface_performance_ratio(self, player_canonical, surface, opponent_canonical, reference_date):
        """Calculate player's surface-specific performance vs opponent's baseline"""
        player_surface_matches = self.historical_data[
            ((self.historical_data['winner_canonical'] == player_canonical) |
             (self.historical_data['loser_canonical'] == player_canonical)) &
            (self.historical_data['Surface'] == surface) &
            (pd.to_datetime(self.historical_data['Date']) <= pd.to_datetime(reference_date))
            ].tail(20)

        opponent_surface_matches = self.historical_data[
            ((self.historical_data['winner_canonical'] == opponent_canonical) |
             (self.historical_data['loser_canonical'] == opponent_canonical)) &
            (self.historical_data['Surface'] == surface) &
            (pd.to_datetime(self.historical_data['Date']) <= pd.to_datetime(reference_date))
            ].tail(20)

        if len(player_surface_matches) < 3 or len(opponent_surface_matches) < 3:
            return 1.0

        player_wins = len(player_surface_matches[player_surface_matches['winner_canonical'] == player_canonical])
        opponent_wins = len(
            opponent_surface_matches[opponent_surface_matches['winner_canonical'] == opponent_canonical])

        player_ratio = player_wins / len(player_surface_matches)
        opponent_ratio = opponent_wins / len(opponent_surface_matches)

        return player_ratio / opponent_ratio if opponent_ratio > 0 else 1.0


# 3.4 Prediction Interface Functions
def convert_to_canonical(name):
    return normalize_name_canonical(name)


def get_matches_for_date(target_date):
    params = {
        "method": "get_fixtures",
        "APIkey": API_KEY,
        "date_start": target_date,
        "date_stop": target_date
    }
    response = SESSION.get(BASE, params=params)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}")

    TOURNAMENT_SURFACES = {
        'ATP Wimbledon': 'Grass',
        'WTA Wimbledon': 'Grass',
        'ATP French Open': 'Clay',
        'WTA French Open': 'Clay',
        'ATP US Open': 'Hard',
        'WTA US Open': 'Hard',
        'ATP Australian Open': 'Hard',
        'WTA Australian Open': 'Hard'
    }

    data = response.json()
    matches = []

    for event in data.get("result", []):
        matches.append({
            'event_key': event.get('event_key'),
            'player1_name': event['event_first_player'],
            'player2_name': event['event_second_player'],
            'tournament_name': event.get('tournament_name', 'Unknown'),
            'tournament_round': event.get('tournament_round', ''),
            'event_status': event.get('event_status', ''),
            'event_type_type': event.get('event_type_type', ''),
            'surface': TOURNAMENT_SURFACES.get(event.get('tournament_name', ''), 'Unknown'),
            'time': event.get('event_time', ''),
            'date': event.get('event_date', '')
        })

    return matches


def get_top_confidence_matches(target_date, top_n=5, min_confidence=0.05):
    matches = get_matches_for_date(target_date)

    results = []
    for match in matches:
        p1_canonical = convert_to_canonical(match['player1_name'])
        p2_canonical = convert_to_canonical(match['player2_name'])

        event_type = str(match.get('event_type_type', '')).lower()
        gender = 'W' if 'wta' in event_type else 'M'
        p1_priors = model.extract_refined_priors(p1_canonical, gender, match['surface'], target_date)
        p2_priors = model.extract_refined_priors(p2_canonical, gender, match['surface'], target_date)

        p1_win_prob = model.simulate_match(p1_priors, p2_priors)
        confidence = abs(p1_win_prob - 0.5)

        if confidence >= min_confidence:
            favorite = match['player1_name'] if p1_win_prob > 0.5 else match['player2_name']
            win_prob = max(p1_win_prob, 1 - p1_win_prob)

            results.append({
                'match': f"{match['player1_name']} vs {match['player2_name']}",
                'favorite': favorite,
                'probability': win_prob,
                'confidence': confidence
            })

    return sorted(results, key=lambda x: x['confidence'], reverse=True)[:top_n]

#%%
# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    global historical_data, jeff_data, weighted_defaults, model

    REFRESH_CACHE = False

    if REFRESH_CACHE:
        print("Refreshing cache...")
        historical_data, jeff_data, weighted_defaults = generate_comprehensive_historical_data(fast=False)
        save_to_cache(historical_data, jeff_data, weighted_defaults)

        days_back = (date.today() - date(2025, 6, 10)).days
        historical_data = integrate_api_tennis_data(historical_data, days_back=days_back)
        save_to_cache(historical_data, jeff_data, weighted_defaults)
    else:
        historical_data, jeff_data, weighted_defaults = load_from_cache()
        if historical_data is None:
            print("Cache miss - generating data...")
            historical_data, jeff_data, weighted_defaults = generate_comprehensive_historical_data(fast=False)
            save_to_cache(historical_data, jeff_data, weighted_defaults)
        else:
            print("✓ Data loaded from cache")

        days_back = (date.today() - date(2025, 6, 10)).days
        historical_data = integrate_api_tennis_data(historical_data, days_back=days_back)
        save_to_cache(historical_data, jeff_data, weighted_defaults)

    # Initialize model
    model = BayesianTennisModel()
    model.historical_data = historical_data
    model.jeff_data = jeff_data

    return historical_data, jeff_data, weighted_defaults, model


if __name__ == "__main__":
    # Initialize global variables
    historical_data, jeff_data, weighted_defaults, model = main()

    # Get today's predictions
    target_date = date.today().isoformat()
    picks = get_top_confidence_matches(target_date, top_n=5, min_confidence=0.05)

    print("\n=== TODAY'S TOP PICKS ===")
    for i, pick in enumerate(picks, 1):
        print(f"{i}. {pick['match']}")
        print(f"   Favorite: {pick['favorite']}")
        print(f"   Win Prob: {pick['probability']:.2%}")
        print(f"   Confidence: {pick['confidence']:.1%}\n")

#%%

# ─── quick integrity checks ──────────────────────────────────────────────
from datetime import date as _d

print("Dataframe shape:", historical_data.shape)

print(
    "Date range:",
    historical_data["date"].min(),
    "→",
    historical_data["date"].max(),
)

print("\nRows by source_rank:")
print(historical_data["source_rank"].value_counts(dropna=False))

after_cutoff = historical_data[
    historical_data["date"] > _d(2025, 6, 10)
]
print("\nRows after 2025‑06‑10:", len(after_cutoff))

if "ta_stats_json" in after_cutoff.columns:
    ta_rows = after_cutoff["ta_stats_json"].notna().mean()
    print("TA scrape coverage after cutoff: {:.1%}".format(ta_rows))

    # preview first few rows with TA data
    if after_cutoff["ta_stats_json"].notna().any():
        cols = [c for c in ("ta_url", "ta_stats_json") if c in after_cutoff.columns]
        print(
            "\nSample TA stats rows:\n",
            after_cutoff.loc[after_cutoff["ta_stats_json"].notna(), cols].head(3),
        )
else:
    print("Column 'ta_stats_json' absent – no TA scrape data stored.")
#%%
