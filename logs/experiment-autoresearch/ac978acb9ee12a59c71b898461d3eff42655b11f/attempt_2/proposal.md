# AutoResearch Proposal

## Target
Experiment-stage modification only.

## Current state summary
Parent state (ac978acb) is a 141-feature `HistGradientBoostingRegressor`
(squared-error loss, RD-weighted training) followed by post-hoc
`IsotonicRegression` fit on a held-out val slice. On the sealed scorer:

- `test_mae` = 232.16 (target 250, **satisfied**; 17.84 Elo headroom).
- `spearman` = 0.665 (target 0.4, **satisfied** comfortably).
- `middle_decile_calibration` = 215.31 (target 100.0, **NOT satisfied**;
  gap 115.31). This is the sole bottleneck.

The internal per-decile table shows tail-compression: the model's
predictions have narrower spread than the truth. D7 (mean-true 1857,
mean-pred 1715) is the internal worst bin in the middle-Elo range; D8
and D9 are worse but outside the [1400, 2000] band the sealed metric
scores.

## Previous attempts from this node
One prior attempt (attempt_1, sha c6f57ac): swapped isotonic **entirely**
for a CDF quantile-matching calibrator. Result on sealed scorer:

- `test_mae` 232.16 → 257.68 → *previously satisfied property lost*, so
  the attempt was **rejected**.
- `middle_decile_calibration` 215.31 → 164.01 (still not satisfied but
  narrowed by ~51 Elo).
- `spearman` 0.665 → 0.665 (monotone transform, preserved).

Whiteboard tip T1 records the same trade-off on internal test:
CDF moved middle-decile calibration 141.7 → 79.0 but MAE 249.2 → 263.5.

Takeaway: pure isotonic under-corrects the tail compression; pure CDF
over-corrects and pays too much MAE. A **blend** should sit on the
Pareto frontier between them.

## Proposed modification
Replace the current single calibrator with a **convex blend** of an
isotonic calibrator and a CDF quantile-matching calibrator, with the
blend weight `alpha` selected on the held-out val slice under an
explicit MAE-headroom constraint.

Concretely, in `src/train.py` after the booster is trained and
`val_pred_raw`, `val_true` are available on the val slice:

1. **Fit both calibrators on val**:
   - Isotonic: `iso = IsotonicRegression(out_of_bounds='clip').fit(val_pred_raw, val_true)`.
   - CDF anchors: `pred_sorted = np.sort(val_pred_raw).astype(np.float32)`,
     `true_sorted = np.sort(val_true).astype(np.float32)`.
2. **Define a callable that applies each and blends**:
   ```
   def calibrate(p, alpha):
       iso_p = iso.predict(p)
       ranks = np.searchsorted(pred_sorted, p, side='left') / len(pred_sorted)
       cdf_p = np.interp(
           ranks,
           np.linspace(0.0, 1.0, len(true_sorted), dtype=np.float32),
           true_sorted,
       )
       cdf_p = np.clip(cdf_p, float(true_sorted[0]), float(true_sorted[-1]))
       return alpha * iso_p + (1.0 - alpha) * cdf_p
   ```
3. **Select `alpha` on val**. To reserve MAE headroom for the sealed
   scorer, do a constrained grid search on
   `alpha_grid = np.linspace(0.0, 1.0, 21)`:
   - Compute a helper val metric `val_mid_dec_cal_1400_2000` that
     mimics the sealed metric: bucket val rows into deciles by true
     rating (only rows whose bin-mean-true rating falls in [1400, 2000]
     are eligible), take max |mean_pred − mean_true| across eligible
     bins.
   - Choose `alpha* = argmin(val_mid_dec_cal_1400_2000)` **subject to**
     `val_mae(alpha) <= max(val_mae(alpha=1.0), 220.0) + 5.0`. That is:
     never allow val MAE to drift more than 5 Elo above the pure-isotonic
     val MAE. This preserves at least ~12 Elo of sealed-MAE headroom
     given the current 17.84 Elo margin, based on the observed
     val↔sealed MAE alignment (val_mae 247.87 in `metrics.json` vs
     sealed 232.16, i.e. sealed is ~16 Elo lower).
   - Break ties toward smaller alpha (more CDF, better calibration).
4. **Persist**: in the joblib artifact, store `iso` (as before),
   `calib_pred_sorted`, `calib_true_sorted`, and the scalar `calib_alpha`.
5. **Inference** (`src/predict.py`): load all four; apply the same
   `calibrate(p, alpha)` transform to raw booster output.
6. **Reproducibility**: rerun the seed-1 twin so the two-seed harness
   (`results/reproducibility.json`) is updated.

**Why this is safe**:
- Both isotonic and CDF calibrators are monotone in raw prediction; a
  convex combination of two monotone functions is monotone, so
  `spearman` is exactly preserved.
- The explicit val-MAE constraint prevents the failure mode of
  attempt_1 (MAE blown past target). The 5-Elo val cushion is chosen
  such that even in the worst-case slack tightening, the sealed MAE
  should remain below 250.
- If the constraint forces `alpha = 1.0`, we recover the parent state
  exactly — no regression is possible.

**Rationale for expecting improvement over parent**:
- Attempt_1 showed sealed calibration moved from 215.31 → 164.01 at
  `alpha=0` with a 25-Elo MAE cost. Under a rough linearity assumption,
  `alpha ~ 0.3–0.5` should land calibration near 180–195 while keeping
  MAE around 240–246 (well inside the 250 target). Even the pessimistic
  case narrows the current 115-Elo gap by ~20–35 Elo.
- MAE(blend) is often *lower* than the linear interpolation of the two
  endpoints when the two calibrators make partially uncorrelated errors
  (standard ensemble effect), so real MAE cost is likely smaller than
  the naive average predicts.

## Keep fixed
- Do not rerun resource finding.
- Do not modify scoring/interface.md.
- Do not read or modify hidden scoring files.
- Keep the research question fixed.
- Keep the booster, features, splits, sample weights, seeds, and the
  `predict(rows)` signature unchanged. Only the calibrator layer changes.

## Expected artifacts to update
- `src/train.py` — fit both calibrators, run the constrained alpha
  search, persist `iso`, `calib_pred_sorted`, `calib_true_sorted`,
  `calib_alpha` into the model artifact.
- `src/predict.py` — load the four calibrator objects and apply the
  blended transform.
- `models/hgbr.joblib` (or equivalent) — regenerated with the new
  calibrator bundle.
- `results/metrics.json`, `results/metrics_seed1.json`,
  `results/reproducibility.json` — regenerated with the blended
  calibrator.
- `results/preds_test.npz`, `results/preds_test_seed1.npz` — regenerated.
- `REPORT.md` — short note in Sections 4.3 (calibration method) and
  5.1/5.3 (updated numbers and selected alpha).

## Success expectation
Public `scoring/results.json` should show:

- `middle_decile_calibration` value moving from 215.31 toward the range
  ~170–200, i.e. narrowing the 115-Elo gap by roughly 15–45 Elo. Full
  closure to 100 is unlikely in one step (that would require widening
  the raw booster predictions, a separate follow-up), but the direction
  of movement should be clearly correct.
- `spearman` unchanged (monotone transform), still ≥ 0.4.
- `test_mae` should remain below 250 by construction of the val
  constraint. Realistic landing zone: 235–248.
- `overall_satisfied` most likely still `false` (calibration bottleneck
  not fully closed), but the bottleneck gap narrows, `test_mae` and
  `spearman` remain satisfied, and no property regresses relative to the
  parent. This sets up a follow-up attempt to attack raw-prediction
  spread (e.g., tail-emphasis sample weights in the booster).
