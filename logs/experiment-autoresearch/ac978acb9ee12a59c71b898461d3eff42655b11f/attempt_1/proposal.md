# AutoResearch Proposal

## Target
Experiment-stage modification only.

## Current state summary
The current pipeline is a 141-feature `HistGradientBoostingRegressor`
(squared-error loss, RD-weighted training) followed by post-hoc
`IsotonicRegression` fit on a held-out val slice. On the sealed scorer,
two of three properties are already satisfied:

- `test_mae` = 232.16 (target 250, satisfied by ~17.8 Elo).
- `spearman` = 0.665 (target 0.4, satisfied comfortably).
- `middle_decile_calibration` = 215.31 (target 100.0, NOT satisfied;
  gap 115.31). This is the sole bottleneck.

The internal per-decile calibration table in `results/metrics.json`
shows the pattern behind the failure: the model's predicted distribution
is significantly narrower than the truth. In the middle-Elo range
[1400, 2000] the largest gap is at D7 (mean-true 1857, mean-pred 1715,
|gap| 141.7). D4/D5/D6 are within 90 Elo. The isotonic calibrator that
follows the booster is monotone but constrained by the raw prediction
range, so it cannot widen the middle-decile spread enough.

## Previous attempts from this node
None. `attempt_history` is empty, whiteboard has no active tips, and the
attempt directory contains only this pending proposal. The public state
reflects the initial scored artifact only.

## Proposed modification
Replace the current `IsotonicRegression` post-hoc calibrator with a
**quantile-matching (CDF) calibrator** fit on the same val slice. Concretely:

1. In `src/train.py` (or wherever the isotonic step lives), after the
   booster is trained, obtain uncalibrated `val_pred` on the val slice
   and the corresponding `val_true` labels.
2. Build two sorted 1-D arrays `pred_sorted = np.sort(val_pred)` and
   `true_sorted = np.sort(val_true)` of equal length. Persist both
   arrays (as float32) inside the joblib artifact alongside the booster,
   e.g. under keys `calib_pred_sorted`, `calib_true_sorted`.
3. Remove the isotonic fit/apply calls.
4. At inference (both in `src/train.py` eval and in `src/predict.py`),
   transform each raw prediction `p` by:
   - `rank = np.searchsorted(pred_sorted, p, side='left') / len(pred_sorted)`
   - `calibrated = np.interp(rank, np.linspace(0, 1, len(true_sorted)),
     true_sorted)`
   - Clip extrapolation to `[true_sorted[0], true_sorted[-1]]`.
5. Preserve the RD-weighted booster, feature set, sample splits, seeds,
   and the `predict(rows)` signature. No changes to features, splits, or
   the booster hyperparameters — this is a calibrator-only swap.
6. Re-run the two-seed reproducibility harness (seed 42, seed 1) to
   confirm cross-seed stability.

Rationale: CDF calibration is a monotone transformation (so `spearman`
is exactly preserved) that explicitly maps the empirical distribution of
raw predictions onto the empirical distribution of true ratings. Unlike
isotonic — which minimizes squared error subject to monotonicity and
therefore compresses predictions where val labels have noisy pockets —
CDF matching restores the full predicted spread, which is precisely
what the middle-decile calibration metric measures. On the internal
D7 bin (mean-true 1857, mean-pred 1715), CDF matching should push
predictions from the 71st predicted-percentile toward the 71st
true-percentile (~1830 Elo), closing most of the ~140 Elo gap. `test_mae`
may drift by a few Elo either way; the current 17.8-Elo margin above the
target absorbs a modest degradation.

## Keep fixed
- Do not rerun resource finding.
- Do not modify scoring/interface.md.
- Do not read or modify hidden scoring files.
- Keep the research question fixed.

## Expected artifacts to update
- `src/train.py` — swap isotonic for CDF calibration; persist the two
  sorted anchor arrays into the model artifact.
- `src/predict.py` — replace isotonic application with the
  `searchsorted` + `interp` transform; load the anchor arrays.
- `models/hgbr.joblib` (or equivalent) — regenerated to include
  `calib_pred_sorted` / `calib_true_sorted` in place of the isotonic
  object.
- `results/metrics.json`, `results/metrics_seed1.json`,
  `results/reproducibility.json` — regenerated with the new calibrator.
- `REPORT.md` — short update to Sections 4.3 (calibration method) and
  5.1/5.3 (numbers + narrative around calibration change).
- `results/preds_test.npz`, `results/preds_test_seed1.npz` — regenerated.

## Success expectation
Public `scoring/results.json` should show:

- `middle_decile_calibration` value moving substantially below its
  current 215.31 toward the 100.0 target. Even a partial closure to
  ~120–140 would represent a large reduction in the sole outstanding
  gap; the internal-test analog is expected to fall from 141.7 toward
  ~50–80 based on the D7/D8 bin shifts that CDF matching produces.
- `spearman` unchanged (monotone transform); still ≥ 0.4 target.
- `test_mae` expected to remain below the 250 target. The change may
  cost a few Elo of MAE because CDF matching does not minimize squared
  error; the current 17.8-Elo slack is sufficient headroom.
- `overall_satisfied` flips to `true` if
  `middle_decile_calibration` reaches 100.0; otherwise the bottleneck
  narrows and the next attempt can iterate on residual middle-Elo bins.
