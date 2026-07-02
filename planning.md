# Research Plan — Predicting Lichess Puzzle Difficulty

## Motivation & Novelty Assessment

### Why This Research Matters
Puzzle-difficulty prediction is a well-calibrated, continuous regression signal on a dataset of 6M+ positions. Solving it reliably enables ex-ante puzzle curation, personalized training, and adaptive tutors without waiting for user-attempt logs to establish an empirical rating.

### Gap in Existing Work
The literature is dominated by two extremes: pure transformer regressors (GlickFormer, MAE 217; requires GPU + days of training) and shallow hand-feature baselines (Aimchess). No compact CPU-only gradient-boosted-tree model with rich tactical features has a public reproducible reference.

### Our Novel Contribution
A compact LightGBM regressor operating on shallow-yet-tactical features derived from FEN + solution sequence, plus theme signals when available. Goal: beat the global-mean baseline (~MAE 430) by a wide margin under strict CPU-only constraints, targeting MAE < 250 on the sealed test slice.

### Experiment Justification
- Baseline (global mean): sanity ceiling; required to demonstrate model added value.
- Per-theme mean: cheap heuristic; establishes that themes carry rating signal.
- Feature-engineered LightGBM (main): tests hypothesis that shallow tactical features (material, mobility, forcing-move counts) plus theme signals suffice to predict rating on CPU-only compute.
- Two seeds for reproducibility check.

## Research Question
Can a compact, CPU-only regressor built on shallow tactical features and theme signals predict Lichess puzzle ratings with test MAE < 250 Elo and Spearman ρ > 0.4?

## Methodology

### Approach
Feature-engineered gradient-boosting on 200–400k puzzles subsampled from the 6M-row corpus, held-out excluding a hash-based validation slice, and finalized as a `predict(rows)` module.

### Experimental Steps
1. Load all 3 parquet shards; sample 400k rows with `RatingDeviation < 90` for label quality.
2. Split train/val/test 70/15/15 by hash of PuzzleId (to mimic the sealed-test rule).
3. Extract features:
   - Position: piece counts by type/color, material balance, side-to-move, castling rights.
   - Solution sequence: length, number of checks, captures, promotions, quiet moves, distance travelled, mate-in-N indicator, ending square activity.
   - Themes: multi-hot encoding of the most common themes (top 30).
   - Opening tag: presence indicator.
4. Baseline 1: global mean.
5. Baseline 2: per-theme mean (average rating over each theme intersected).
6. Main: LightGBM regressor (~800 trees, depth 8). Train with Huber loss for robustness to RD-noise.
7. Evaluate MAE, RMSE, Spearman, per-decile calibration on held-out test split.
8. Robustness: retrain with seed=1; report ΔMAE.

### Baselines
- Global mean rating.
- Per-theme mean rating.

### Evaluation Metrics
- **MAE** (primary): interpretable, matches literature.
- **RMSE** (secondary): compatible with IEEE BigData Cup metric.
- **Spearman ρ**: rank-quality check.
- **Middle-decile calibration**: |mean(pred) - mean(true)| on rating bins 1400-2000.

### Statistical Analysis Plan
Two-seed rerun (seed 42, 1) to report ΔMAE stability. Bootstrap 1000 samples of test MAE for a confidence interval.

## Expected Outcomes
- Main model MAE 200-240 (informed by GlickFormer transformer at MAE 217 and BigData Cup GBDTs at ~205-215).
- Spearman ρ > 0.7.
- Robust to seed shifts within ±10 MAE.

## Timeline
- Phase 1 planning: 10 min
- Phase 2 data load + feature build: 40 min
- Phase 3 baseline + training + eval: 60 min
- Phase 4 predict.py packaging: 20 min
- Phase 5 analysis + figs: 30 min
- Phase 6 report: 30 min

## Potential Challenges
- Feature extraction on 400k puzzles must complete in <20 min → keep feature functions pure-Python + numpy; parallelize with joblib if needed.
- LightGBM can overfit to themes if we're not careful; use RD-clipped labels and cap tree depth.
- Model artifact must load quickly at import time; store as small pickled sklearn/lightgbm booster.
- Themes may be missing at inference; must degrade gracefully.

## Success Criteria
- Test MAE < 250 (hypothesis threshold).
- Spearman ρ > 0.4.
- Middle-decile calibration error < 100.
- Cross-seed ΔMAE < 20.

## Notes on Scoring Protocol
- `src/predict.py` must expose `predict(rows: list[dict]) -> list[float]`.
- Do not use `NbPlays`, `RatingDeviation`, `Popularity` as inputs.
- Model must load at import time or lazily on first call from files under `src/` or `models/`.
- Must be deterministic; no randomness at inference.
- Handle missing/empty `Themes` gracefully.
