# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tennis prediction system with 3 weeks of development. Functional data pipeline processing ~30k matches from multiple sources, but significantly underutilizing available data richness. Currently achieving undergraduate-level modeling despite having PhD-level data.

## Current Architecture Status

### Data Sources & Volume
- **Jeff Sackmann Data**: 25k matches (2020-2025) with complete point-by-point sequences
- **API-Tennis**: Standard tier covering post-2025/06/10 matches (no live points/odds access)
- **Tennis Abstract**: Automated scraping for recent matches with detailed charting
- **Excel Files**: Historical match results pre-2020

### Core Strengths
- Robust player canonicalization with bidirectional mapping
- Content-based caching with compression (SHA-256 hashing)
- Async API integration with rate limiting
- Hierarchical data source prioritization (TA > API > Excel)
- Production-ready error handling and logging

### Critical Underutilizations

#### 1. Jeff's Point Sequences - HIGHEST PRIORITY
**Have**: `chartingmpoints*.csv` with `1st`/`2nd` columns containing shot-by-shot notation
```
Example: '4f8b3f*' = wide serve, forehand crosscourt, backhand line, forehand winner
```
**Using**: Nothing - completely ignored
**Should Extract**: Serve patterns, rally lengths, shot directions, momentum states

#### 2. Static Aggregations Instead of Distributions
**Have**: Set-by-set statistics, conditional performance data
**Using**: Simple means (`winner_aces = 6`)
**Should Compute**: Variance, conditional stats (serve% under pressure), full distributions

#### 3. No Sequential/Temporal Modeling
**Have**: Point-by-point progressions with score states
**Using**: Match-level aggregates only
**Missing**: Momentum tracking, form decay, fatigue indicators

#### 4. Primitive Feature Engineering
Current features are basic ratios. Missing:
- Player interaction modeling (style matchups)
- Surface transition penalties
- Time-varying parameters
- Uncertainty quantification

## Immediate Implementation Priorities

### 1. Parse Jeff's Shot Notation
```python
def parse_shot_sequence(point_str):
    """
    Parse Jeff Sackmann's notation system
    First char: 4=wide, 5=body, 6=T
    Middle chars: f=forehand, b=backhand, r=rally, v=volley
    Direction: 1-9 (court zones)
    Last char: *=winner, @=unforced error, #=forced error
    """
```

### 2. Build Transition Matrices
```python
# Serve location → return depth → point outcome
serve_return_matrix[serve_loc][return_depth] = win_probability

# Rally patterns
rally_transitions[prev_shot][next_shot] = frequency
```

### 3. Add Temporal Features
```python
# Form decay
matches['form_weight'] = np.exp(-0.1 * matches['days_ago'])

# Fatigue indicators
matches['fatigue_index'] = matches_last_14d * avg_match_duration

# Surface transitions
matches['surface_switch_penalty'] = days_since_surface_change
```

### 4. Extract Momentum Features
```python
# From existing point data
consecutive_points = identify_winning_streaks(point_sequence)
pressure_situations = extract_break_points(score_states)
comeback_patterns = detect_momentum_shifts(game_scores)
```

## Technical Debt & Issues

### Data Pipeline
- Deduplication occurs AFTER one-hot encoding (potential label leakage)
- Date handling inconsistencies between data sources
- No validation of winner/loser assignment in scraped data

### Model Architecture
- Fixed ensemble weights instead of dynamic weighting
- No uncertainty quantification (point estimates only)
- Missing player embeddings or interaction terms
- No handling of cold-start (new players)

### Production Gaps
- No backtesting framework with proper temporal splits
- Missing calibration metrics
- No continuous learning pipeline
- Limited feature computation caching

## Ideal Architecture Vision

### Phase 1: Enhanced Feature Engineering
1. **Sequence Features**: Parse all Jeff notation, build Markov chains
2. **Temporal Features**: Implement decay, fatigue, form streaks
3. **Contextual Features**: Style embeddings, H2H evolution, pressure stats

### Phase 2: Advanced Modeling
1. **Bayesian Hierarchical Model**: Player skills with surface/opponent adjustments
2. **Neural Sequence Model**: LSTM/Transformer for momentum evolution
3. **Gradient Boosting**: Custom objectives incorporating betting ROI
4. **Dynamic Ensemble**: Weight models by recent performance

### Phase 3: Production System
```python
class TennisPredictionService:
    - Real-time feature extraction
    - Uncertainty quantification
    - Market calibration
    - Continuous learning updates
```

## File Structure Enhancement Needed

```
project/
├── features/
│   ├── sequence_parser.py      # Jeff notation parsing
│   ├── momentum_extractor.py   # HMM for game states
│   ├── temporal_features.py    # Form, fatigue, transitions
│   └── interaction_features.py # Player matchups
├── models/
│   ├── bayesian_model.py      # Hierarchical Bayesian
│   ├── sequence_model.py      # LSTM/Transformer
│   ├── ensemble_dynamic.py    # Adaptive weighting
│   └── calibration.py         # Probability calibration
├── evaluation/
│   ├── backtesting.py         # Temporal validation
│   ├── betting_roi.py         # ROI simulation
│   └── calibration_metrics.py # Reliability diagrams
└── production/
    ├── feature_cache.py       # Redis/persistent cache
    ├── model_registry.py      # Version management
    └── continuous_learning.py # Incremental updates
```

## Development Guidelines

### When Adding Features
1. Always check if data already exists in Jeff's sequences before computing
2. Prefer conditional distributions over simple averages
3. Include temporal decay in any historical calculation
4. Add uncertainty estimates to all predictions

### When Training Models
1. Use time-based splits (no future data leakage)
2. Weight recent matches more heavily
3. Validate on multiple metrics (accuracy, calibration, ROI)
4. Save feature importance for interpretability

### Performance Optimization
- Use Polars for large CSV operations
- Cache computed features aggressively
- Batch API requests with async processing
- Profile memory usage for Jeff data processing

## Testing Requirements

### Unit Tests Needed
- Jeff notation parser validation
- Momentum state extraction accuracy
- Feature computation correctness
- Model calibration metrics

### Integration Tests Needed
- Full pipeline with all data sources
- Production prediction workflow
- Cache invalidation scenarios
- New player handling

## Configuration

### Environment Variables
```bash
API_TENNIS_KEY=your_key_here
TENNIS_DATA_DIR=~/Desktop/data
TENNIS_CACHE_DIR=~/Desktop/data/cache
JEFF_DATA_DIR=~/Desktop/data/Jeff 6.14.25
```

### Key Settings (settings.py)
- `BASE_CUTOFF_DATE`: 2025-06-10 (Jeff data boundary)
- `CACHE_COMPRESSION_LEVEL`: 3 (zlib compression)
- `MAX_CONCURRENT_REQUESTS`: 10 (API rate limiting)
- `FEATURE_EXTRACTION_BATCH_SIZE`: 10000

## Next Sprint Priorities

1. **Implement Jeff notation parser** - 20%+ immediate improvement
2. **Add momentum HMM** - Critical for in-match dynamics
3. **Build form decay features** - Better recent performance weighting
4. **Create Bayesian model** - Uncertainty quantification
5. **Develop backtesting framework** - Proper evaluation

## Notes for Claude Code

- The infrastructure is solid but the statistical modeling needs elevation
- Focus on extracting signal from existing data before adding new sources  
- Jeff's point sequences are the highest-value untapped resource
- Temporal dynamics are completely missing from current implementation
- The model ensemble uses fixed weights when it should adapt to data availability