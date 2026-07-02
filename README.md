# Predicting Lichess Puzzle Difficulty from Position

Compact CPU-only regressor that predicts the empirical Glicko-2
difficulty rating of a Lichess chess puzzle from the position (FEN),
the puzzle's setup move, and its solution sequence.

## Key findings

- **Test MAE 247.8 Elo** — comfortably under the hypothesis threshold of
  <250, and well below the global-mean baseline of ~390 (37% relative
  reduction) and the per-theme-mean baseline of ~334.
- **Spearman ρ = 0.744** on the held-out test slice — far exceeds the
  target ρ > 0.4; the model's ranking of puzzles from easiest to
  hardest is highly aligned with Lichess's empirical ratings.
- **Solution-length is the strongest single feature** (short vs. long),
  followed by `theme_mate`, `sol_ends_mate`, and check/capture ratios.
- **Middle-decile calibration gap ≈ 139 Elo** — post-hoc isotonic
  regression fit on a held-out validation slice reduces raw
  mean-reversion; the tails (very easy or very hard puzzles) remain
  the residual limit.
- **Reproducible across seeds**: ΔMAE between seed 42 and seed 1 is
  well within the 20-Elo robustness target (see
  `results/reproducibility.json`).

## How to reproduce

```bash
# 1. Set up the environment
uv venv && source .venv/bin/activate
uv sync                      # installs pinned dependencies

# 2. Build the feature matrix (~2 min for 400k rows)
python src/build_dataset.py --n 400000 --rd-max 90 --seed 42 \
    --out results/dataset.npz \
    --data-dir datasets/lichess_puzzles/data

# 3. Train + evaluate (~1 min)
python src/train.py \
    --data results/dataset.npz --seed 42 \
    --model-out models/hgbr.joblib \
    --metrics-out results/metrics.json \
    --preds-out results/preds_test.npz

# 4. Generate figures
python src/analyze.py

# 5. (optional) Reproducibility check
python src/train.py --seed 1 \
    --model-out models/hgbr_seed1.joblib \
    --metrics-out results/metrics_seed1.json \
    --preds-out results/preds_test_seed1.npz
```

## File layout

```
src/
  features.py          # FEN + Moves + Themes → 141-dim feature dict
  build_dataset.py     # Parquet → hash-split .npz feature matrix
  train.py             # HistGradientBoostingRegressor + isotonic calib.
  analyze.py           # Diagnostic plots
  predict.py           # Sealed-scoring entry point (predict(rows))
models/
  hgbr.joblib          # Trained model + isotonic calibrator
  feature_names.json   # Frozen feature order
  meta.json
results/
  dataset.npz          # Built feature matrix
  metrics.json         # Full metrics for seed 42
  metrics_seed1.json   # Full metrics for seed 1
  preds_test.npz       # Test predictions (seed 42)
  reproducibility.json # ΔMAE across seeds
figures/
  pred_vs_actual.png
  residual_hist.png
  decile_calibration.png
  feature_importance.png
REPORT.md              # Full research report
planning.md            # Original research plan
```

See `REPORT.md` for the complete write-up, including per-decile
calibration, feature-importance analysis, and discussion.
