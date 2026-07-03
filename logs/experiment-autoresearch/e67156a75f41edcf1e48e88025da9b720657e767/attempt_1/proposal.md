# AutoResearch Proposal

## Target
Experiment-stage modification only.

## Current state summary
The parent state ships a 141-feature HistGradientBoostingRegressor with a
convex-blend calibrator (isotonic + CDF quantile-matcher, alpha=0.35 chosen on
val under a val-MAE budget). Public scoring/results.json reports:

- `test_mae` = 244.31 (target <250, **satisfied**, headroom ~5.7 Elo).
- `spearman` = 0.666 (target >0.4, **satisfied**).
- `middle_decile_calibration` = 174.00 (target <100, **NOT satisfied**, gap 74.0).
- `overall_satisfied` = false; declared bottleneck: `middle_decile_calibration`.

Internally the model achieves mid-decile-cal 100.9 on the 50k held-out slice, but
the sealed slice (n=2000, RD<=80, middle bins [1400,2000]) exposes a ~74 Elo gap.
Per-decile analysis in REPORT.md shows the raw booster still under-predicts on
D7-D9 (D7 pred 1756 vs true 1857; D8 1863 vs 2045; D9 2013 vs 2341). The
calibrator has already pulled most of the achievable tail spread out of the raw
predictions; further gains must come from widening the booster's raw predictions
on hard puzzles.

## Previous attempts from this node
No prior attempts exist at this node.

## Proposed modification
Add **tail-emphasis training sample weights** to the HGBR fit in `src/train.py`
so the booster puts more capacity on hard-puzzle predictions in the Elo band the
sealed scorer measures. Concretely, multiply the existing RD-based weight by a
piecewise-linear tail factor keyed on `y_train`:

```
def tail_weight(y):
    # 1.0 for y <= 1700
    # ramp 1.0 -> 1.8 across [1700, 2100]
    # 1.8 for y in [2100, 2400]
    # ramp 1.8 -> 1.0 across [2400, 2700]  (avoid over-fitting extreme tails)
    # 1.0 for y > 2700
    return np.interp(y, [1700, 2100, 2400, 2700], [1.0, 1.8, 1.8, 1.0])

sample_weight = rd_weight * tail_weight(y_train)
```

Everything else stays the same: same features, same 300k/50k/50k split, same
HGBR hyperparameters (`squared_error`, `lr=0.07`, `max_leaf_nodes=127`,
`min_samples_leaf=100`, `l2=0.5`, `max_iter=500`, early stopping), same convex
isotonic+CDF calibrator with the alpha grid search and the `val_mae(alpha) <=
max(val_mae(1.0), 220) + 5` constraint. The calibrator will be re-fit on the
new booster's val predictions and will pick its own alpha.

Rationale (tip T3): the calibrator sits on its Pareto frontier and cannot
create resolution the raw booster lacks. Upweighting rows with y_true in
[1700, 2400] shifts squared-error minima upward in that band, widening the raw
predicted spread on D7-D9 exactly where the sealed scorer is unsatisfied.
Because the calibrator is monotone and refit on val, Spearman is preserved and
the val-MAE headroom check protects against a large MAE regression. If tail
upweighting inflates val MAE beyond the alpha budget, the fallback to alpha=1.0
keeps behavior no worse than pure isotonic on the new booster.

Sanity guards inside the change:
- Cap the tail-weight factor at 1.8x to avoid destabilizing early stopping.
- Rerun seed=1 as before and update `results/reproducibility.json`.
- Regenerate `figures/decile_calibration.png` and refresh
  `results/metrics.json`.

## Keep fixed
- Do not rerun resource finding.
- Do not modify scoring/interface.md.
- Do not read or modify hidden scoring files.
- Keep the research question fixed.
- Keep the feature set, splits, HGBR hyperparameters, and calibrator architecture
  unchanged; only the training sample_weight formula changes.
- Keep `src/predict.py`'s public `predict(rows)` signature and no-network,
  deterministic contract.

## Expected artifacts to update
- `src/train.py` (tail-weight factor added to `sample_weight` construction).
- `models/hgbr.joblib`, `models/meta.json` (retrained booster + refit
  calibrator with newly selected alpha).
- `results/metrics.json`, `results/metrics_seed1.json`,
  `results/reproducibility.json`, `results/preds_test.npz`,
  `results/preds_test_seed1.npz`.
- `REPORT.md` per-decile table and analysis section.
- `figures/decile_calibration.png`, `figures/pred_vs_actual.png`,
  `figures/residual_hist.png` (regenerated).

## Success expectation
On scoring/results.json the primary target is to lower
`middle_decile_calibration` from 174.00 toward the <100 threshold by widening
the raw booster's predictions in the [1800, 2200] band that dominates the
sealed middle-Elo bins. `spearman` should remain >0.4 (monotone calibrator
preserves rank order exactly on the sealed slice; expected to stay near the
parent's 0.666). `test_mae` should remain <250: internally the RD-weighted
squared-error fit has ~5.7 Elo of sealed headroom, and the calibrator's
alpha-selection budget explicitly reserves it; if tail upweighting nudges val
MAE up, alpha shifts toward 1.0 (pure isotonic) automatically, bounding
MAE regression. Realistic outcome: sealed
`middle_decile_calibration` drops materially (target: <=120, stretch: <100);
`test_mae` stays in the 244-250 band; `overall_satisfied` becomes true if the
mid-decile gap crosses 100.
