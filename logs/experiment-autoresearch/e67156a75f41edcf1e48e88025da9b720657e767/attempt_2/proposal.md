# AutoResearch Proposal

## Target
Experiment-stage modification only.

## Current state summary
Parent state (e67156a) ships a 141-feature HistGradientBoostingRegressor plus a
convex blend of an isotonic calibrator and a CDF quantile-matcher, with alpha
selected on the full val slice under a val-MAE budget. Public
`scoring/results.json` reports:

- `test_mae` = 244.31 Elo (target <250, **satisfied**, headroom ~5.7 Elo).
- `spearman` = 0.666 (target >0.4, **satisfied**).
- `middle_decile_calibration` = 174.00 (target <100, **NOT satisfied**, gap 74.0).
- `overall_satisfied` = false; declared bottleneck: `middle_decile_calibration`.

The public score meta records `rd_max = 80`, `middle_elo_range = [1400, 2000]`,
`n_test = 2000`. Internally the same pipeline reaches mid-decile-cal 100.9 on the
50k-test slice (essentially at the <100 stretch target), so the sealed gap of ~73
Elo is not driven by insufficient raw calibrator capacity but by a distribution
mismatch between the val slice used to pick alpha (all puzzles with RD<90) and
the sealed slice (RD<=80, filtered to middle bins).

## Previous attempts from this node
One attempt (7f7ec7a) has already been made and REJECTED. It added a
piecewise-linear tail-emphasis factor
(`interp(y, [1700, 2100, 2400, 2700], [1.0, 1.8, 1.8, 1.0])`) to the training
`sample_weight`. Public score movements:

- `middle_decile_calibration`: 174.00 -> 176.68 (regressed, still the bottleneck).
- `test_mae`: 244.31 -> 248.02 (still under 250, less headroom).
- `spearman`: 0.6656 -> 0.6598 (still above 0.4).

Whiteboard tip T4 records why the trick did not help: internally the mid-decile
metric is dominated by D7 (mean-true 1857), which tail upweighting only widens
raw preds for by ~2-4 Elo, and the val MAE budget then constrains alpha to the
same 0.35 the parent already picks. The regressor's raw resolution in the
[1400, 2000] band is not the binding constraint; the calibrator's
alignment with the sealed slice distribution is.

## Proposed modification
Modify **only the calibrator-alpha selection procedure inside `src/train.py`**
so it optimizes against a val subset that mimics the sealed slice's noise level.

Concretely, in the block that currently searches
`np.linspace(0, 1, 21)` for the alpha that minimizes the middle-decile
calibration proxy under a val-MAE budget:

1. Build a filtered view of val:
   ```python
   val_mask = rd_val <= 80         # matches sealed rd_max
   y_val_f  = y_val[val_mask]
   raw_val_f = raw_val[val_mask]
   iso_val_f = iso(raw_val_f)
   cdf_val_f = cdf(raw_val_f)
   ```
   (`rd_val` is already available - it feeds the RD-weight during training and
   is preserved on the val split. If it is not currently carried through, add it
   to the val cache in `build_dataset.py`'s output or reload it from the parquet
   using the val row indices.)
2. Compute the mid-decile calibration proxy on `y_val_f` using the same decile
   scheme (10 equal-frequency bins of `y_val_f`, keep bins whose mean-true is in
   [1400, 2000], take max |mean_pred - mean_true|). This mirrors the sealed
   evaluator's filter.
3. Compute the val-MAE budget on the SAME filtered slice:
   `mae_budget = max(mae_iso_on_val_f, 220) + 5`. Filtering may shift the raw
   iso MAE by a few Elo; scoring headroom is preserved because sealed MAE is
   evaluated on a similarly RD-filtered population.
4. Widen the alpha grid to `np.linspace(0.0, 1.0, 41)` (step 0.025) for finer
   resolution near the elbow. Keep the existing tiebreak toward smaller alpha
   and the fallback `alpha = 1.0` when no grid point is feasible.

Everything else stays the same: same features, same 300k/50k/50k hash split,
same HGBR hyperparameters, same isotonic + CDF construction, same
val-MAE-budget-with-fallback contract, same `predict.py` signature and
determinism. Only the val slice used to pick alpha changes.

Rationale: whiteboard tip T2 established that the isotonic+CDF blend at
alpha=0.35 sits on the calibrator's Pareto frontier with respect to the
*full val* proxy. But the sealed scorer restricts to RD<=80 puzzles in
[1400, 2000], where the label noise is much lower. On the noisier full-val
proxy, small alpha regions look bad because CDF stretches predictions into
RD-noisy tails; on a RD<=80 val subset, that same stretch aligns with cleaner
truth values and the true minimum shifts to a slightly smaller alpha
(likely 0.20-0.30). Because the calibrator is still monotone in the raw
prediction, Spearman stays above 0.4 by construction, and the val-MAE budget
still guards against a MAE blowup.

Guards inside the change:
- If the RD<=80 mask reduces val below 20k, back off to `rd <= 85` before
  falling back to unfiltered val, so we do not lose statistical power.
- Keep `alpha_fallback = 1.0` behavior if the constraint is infeasible.
- Retrain and re-fit are cheap because only the alpha search changes; the
  booster and calibrators themselves do not need retraining if they are
  already cached. Only `results/metrics.json`, `results/metrics_seed1.json`,
  `models/meta.json`, and the calibrated predictions need refreshing.

## Keep fixed
- Do not rerun resource finding.
- Do not modify `scoring/interface.md`.
- Do not read or modify hidden scoring files
  (`scoring/eval.py`, `scoring/targets.json`, `.scoring_sealed/`).
- Keep the research question fixed.
- Keep the feature set, splits, HGBR hyperparameters, and calibrator
  architecture (blend of monotone isotonic + monotone CDF) unchanged.
- Keep `src/predict.py`'s public `predict(rows)` signature and its
  no-network, deterministic contract.
- Keep the fallback to `alpha = 1.0` when the val-MAE budget is infeasible so
  the pipeline can never regress below pure-isotonic behaviour.

## Expected artifacts to update
- `src/train.py` (alpha-selection block updated to use the RD<=80 val subset
  and the 41-point grid).
- `src/build_dataset.py` if `rd_val` is not already exposed on the val split;
  otherwise unchanged.
- `models/meta.json` (records the newly selected alpha and the filtered
  val-MAE / mid-decile proxy values).
- `results/metrics.json`, `results/metrics_seed1.json`,
  `results/reproducibility.json`, `results/preds_test.npz`,
  `results/preds_test_seed1.npz` (recomputed with the newly selected alpha).
- `figures/decile_calibration.png`, `figures/pred_vs_actual.png`,
  `figures/residual_hist.png` (regenerated).
- `REPORT.md` per-decile table + calibration discussion updated to reflect
  the RD-filtered alpha-selection procedure and the new alpha value.

## Success expectation
The primary lever is `middle_decile_calibration` on `scoring/results.json`.
By selecting alpha on a val subset whose RD distribution matches the sealed
scorer's `rd_max = 80`, alpha should move toward the CDF end (smaller value),
which widens raw prediction spread in the [1400, 2000] band that dominates the
sealed measurement. Because both isotonic and CDF are monotone in the raw
prediction, their convex blend is monotone, so `spearman` stays exactly at
its rank order (expected to remain near the parent's 0.666, well above 0.4).
The val-MAE budget still constrains alpha so `test_mae` should stay under 250:
the sealed slice is RD<=80 and mid-Elo, which is exactly the population the
filtered budget guards against; if the constraint becomes infeasible the
fallback to `alpha = 1.0` restores the parent's isotonic-only behavior
(sealed MAE 244.31, mid-cal 174.00) so regression relative to the parent is
bounded by design.

Realistic outcome: sealed `middle_decile_calibration` drops from 174.00
toward the <100 threshold (target: <=130 conservative, <100 stretch);
`test_mae` stays in the 244-250 band; `spearman` stays near 0.666;
`overall_satisfied` flips to true if the mid-decile gap crosses 100.
