# AutoResearch Proposal

## Target
Experiment-stage modification only.

## Current state summary
Parent state (e67156a) ships a 141-feature HistGradientBoostingRegressor with a
convex-blend calibrator (isotonic + CDF quantile-matcher, alpha=0.35 chosen on
val under a val-MAE budget). Public `scoring/results.json` reports:

- `test_mae` = 244.31 (target <250, **satisfied**, headroom ~5.7 Elo).
- `spearman` = 0.666 (target >0.4, **satisfied**).
- `middle_decile_calibration` = 174.00 (target <100, **NOT satisfied**, gap 74).
- `overall_satisfied` = false; declared bottleneck: `middle_decile_calibration`.

Public score meta: `rd_max = 80`, `middle_elo_range = [1400, 2000]`,
`n_test = 2000`.

The parent REPORT.md diagnoses the residual mid-decile gap explicitly: *"the
underlying booster genuinely cannot separate hard puzzles in feature space; a
calibrator can only redistribute mass, not create new resolution."* The per-decile
table in results/metrics.json confirms it: D7 raw pred 1756 vs true 1857, D8 1863
vs 2045, D9 2013 vs 2341. The booster's raw prediction spread in the middle Elo
band [1400, 2000] is what limits calibration; the calibrator has already extracted
most of the achievable tail spread from the current 141-dim feature vector.

## Previous attempts from this node
Two attempts have already been made from this parent node, both REJECTED:

- **Attempt 1 (7f7ec7a): tail-emphasis sample weights.** Added a piecewise-linear
  `tail_weight(y)` factor multiplied into the RD-based training weight. Public
  score moves: `middle_decile_calibration` 174.00 -> 176.68 (**regressed**),
  `test_mae` 244.31 -> 248.02, `spearman` 0.666 -> 0.660. Whiteboard tip T4
  explains the failure: the sealed mid-decile metric is dominated by D7 (mean-true
  1857), and tail upweighting only widens raw preds for that band by ~2-4 Elo,
  which is absorbed by the val-MAE budget that keeps alpha at 0.35.
- **Attempt 2 (d3e7fff): RD-filtered val for calibrator-alpha selection.** Modified
  `src/train.py` to pick alpha on val subset RD<=80 and widened the alpha grid.
  Public score moves: `middle_decile_calibration` 174.00 -> 195.73
  (**much worse**), `test_mae` 244.31 -> 235.49 (better), `spearman` 0.666 -> 0.666.
  Whiteboard tip T5 predicted this: filtering val to RD<=80 lowers iso val-MAE by
  ~15 Elo, which tightens the MAE budget and pushes alpha UP to ~0.70 (more
  isotonic, less CDF spread), narrowing raw prediction spread on mid-Elo bins.

Together these tell us: attacks on the calibrator alpha AND on the sample-weight
formulation are both constrained by the same val-MAE budget and both hit the
Pareto frontier of the current 141-feature booster. The next-attempt axis must
be to give the raw booster more information about mid-difficulty puzzles.

## Proposed modification
Add **tactical-density features** to `src/features.py` and rebuild the dataset,
then retrain the HGBR and refit the existing isotonic+CDF blend calibrator. The
proposed additions are cheap, deterministic, and computed with the same
python-chess primitives already in use. They target the D7-D9 resolution gap
directly by encoding "how many candidate moves does the solver face at each ply"
and "does the setup involve a material sacrifice", both of which are strong
first-order signals for medium-to-hard puzzle difficulty and are not currently
captured by the sol_len / sol_check_ratio family.

Concretely, add the following feature block to `src/features.py`, executed on
the same board object that already walks the solution sequence:

```python
# For each ply of the solution, record:
#   - number of legal moves (candidate breadth)
#   - number of legal captures (tactical density)
#   - number of legal checks (forcing-move density)
# Then reduce across the sequence into a small set of scalar features.

legal_counts = []
legal_captures = []
legal_checks = []
for ply_board in walk_solution(board, moves):
    legals = list(ply_board.legal_moves)
    legal_counts.append(len(legals))
    legal_captures.append(sum(1 for m in legals if ply_board.is_capture(m)))
    # cheap check test: push/pop
    n_checks = 0
    for m in legals:
        ply_board.push(m)
        if ply_board.is_check():
            n_checks += 1
        ply_board.pop()
    legal_checks.append(n_checks)

feats['sol_legal_max']       = max(legal_counts)     if legal_counts else 0
feats['sol_legal_mean']      = mean(legal_counts)    if legal_counts else 0
feats['sol_legal_last']      = legal_counts[-1]      if legal_counts else 0
feats['sol_legal_sum']       = sum(legal_counts)
feats['sol_captures_max']    = max(legal_captures)   if legal_captures else 0
feats['sol_captures_mean']   = mean(legal_captures)  if legal_captures else 0
feats['sol_captures_sum']    = sum(legal_captures)
feats['sol_checks_max']      = max(legal_checks)     if legal_checks else 0
feats['sol_checks_mean']     = mean(legal_checks)    if legal_checks else 0
feats['sol_checks_sum']      = sum(legal_checks)

# Setup-move features (the opponent's first move, which is applied to the FEN
# to produce the puzzle-start position). Already parsed as `setup_move` in the
# code; extract these signals from it:
feats['setup_is_capture']    = int(pre_board.is_capture(setup_move))
feats['setup_gives_check']   = int(setup_gives_check_flag)  # already available
feats['setup_piece_type']    = piece_type_id(pre_board, setup_move.from_square)
feats['setup_to_center']     = center_distance(setup_move.to_square)  # cheap
```

Everything else stays the same: same 300k/50k/50k hash split, same HGBR
hyperparameters (`squared_error`, `lr=0.07`, `max_leaf_nodes=127`,
`min_samples_leaf=100`, `l2=0.5`, `max_iter=500`, early stopping), same
RD-based training sample_weight (parent's `1/(1+RD/40)`), same convex isotonic +
CDF calibrator with parent's alpha grid on the full (RD<=90) val slice and the
val-MAE budget `max(val_mae(alpha=1.0), 220) + 5`. The calibrator will refit
naturally on the new booster's val predictions and select its own alpha.

Rationale: the sealed slice's D7-D9 puzzles have `mean_true` at 1857 / 2045 /
2341, and the parent's raw predictions on them are 1756 / 1863 / 2013 —
compressed by ~100 / 180 / 330 Elo. The current features (sol_len, sol_check_ratio,
mate flags, theme flags) heavily rank first-order tactics (short mates, one-move
solutions) but do not encode the tactical density of positions with many
plausible moves — which is exactly what separates a 1500-rated puzzle from a
2000-rated puzzle. Adding `sol_legal_*`, `sol_captures_*`, `sol_checks_*`
gives the booster a direct signal for candidate-move ambiguity, and the setup
features (capture, check, piece type, centrality) let it distinguish "quiet
setup" puzzles from "opponent-forced-move" puzzles that tend to be harder. All
of these are position-derived (not label-leaking), deterministic, and require
no external engine.

Guards inside the change:
- Add the new features at the end of the feature vector so the training/val/test
  arrays remain positionally compatible with the retrain pipeline.
- Update `models/feature_names.json` to reflect the new dim; the frozen name list
  is how `src/predict.py` aligns inference features to the trained model.
- Retrain seed=42 AND seed=1 (existing reproducibility protocol) so
  `results/reproducibility.json` reports the new cross-seed deltas.
- Regenerate figures and REPORT.md per-decile table.
- Keep the val-MAE-budget-with-fallback contract; if the new booster's alpha
  search returns infeasible, fall back to `alpha=1.0` (pure isotonic) — same
  no-regression guarantee as parent.
- Compute budget: each new feature is O(legal_moves) per ply, so per puzzle it's
  roughly 5-10x the current move-walk cost; on 300k rows this should stay well
  under the parent's dataset-build time.

## Keep fixed
- Do not rerun resource finding.
- Do not modify `scoring/interface.md`.
- Do not read or modify hidden scoring files
  (`scoring/eval.py`, `scoring/targets.json`, `.scoring_sealed/`).
- Keep the research question fixed.
- Keep the 300k/50k/50k hash-based split, the RD<=90 training filter, and the
  RD-based training sample weight formula unchanged.
- Keep HGBR hyperparameters (`squared_error`, `lr=0.07`, `max_leaf_nodes=127`,
  `min_samples_leaf=100`, `l2=0.5`, `max_iter=500`, early stopping) unchanged.
- Keep the calibrator architecture (convex blend of monotone isotonic + monotone
  CDF, alpha selected on RD<=90 val under val-MAE budget with alpha=1.0
  fallback) unchanged.
- Keep `src/predict.py`'s public `predict(rows)` signature and its no-network,
  deterministic contract.
- Do not use `NbPlays`, `RatingDeviation`, or `Popularity` as inputs to the new
  features (the RD-weight during training is not an input feature).

## Expected artifacts to update
- `src/features.py` (new feature functions and additions to the feature dict).
- `src/build_dataset.py` if needed to plumb any new fields through the
  extraction pipeline; otherwise unchanged.
- `models/hgbr.joblib` (retrained on the expanded feature vector).
- `models/feature_names.json` (extended with the new feature names in the same
  order as `src/features.py` emits them).
- `models/meta.json` (new dim, refit calibrator alpha, refreshed val metrics).
- `results/dataset.npz` (rebuilt feature matrix cache).
- `results/metrics.json`, `results/metrics_seed1.json`,
  `results/reproducibility.json`, `results/preds_test.npz`,
  `results/preds_test_seed1.npz`.
- `figures/decile_calibration.png`, `figures/pred_vs_actual.png`,
  `figures/residual_hist.png`, `figures/feature_importance.png` (regenerated).
- `REPORT.md` — per-decile table, feature-importance table, and analysis
  discussing whether tactical-density features close the D7-D9 gap.

## Success expectation
The primary target on `scoring/results.json` is `middle_decile_calibration`,
which must drop from 174.00 toward the <100 threshold. New tactical-density
features give the booster direct signals to separate D7-D9 puzzles from D5-D6
puzzles — the candidate-count and capture/check density features are precisely
the axes that distinguish mid-Elo tactical complexity, and they are absent from
the parent's 141-dim vector. Because the calibrator is refit on new val
predictions, whatever residual mean-reversion remains will be corrected by the
existing isotonic+CDF blend, whose alpha search is unchanged and whose fallback
to alpha=1.0 prevents regression below pure-isotonic behavior.

`spearman` should remain well above 0.4: monotone calibrators preserve rank
order, and richer features can only increase raw booster discrimination.
Expected to stay near or above the parent's 0.666.

`test_mae` should remain below 250: the new features are strict additions with
no change to loss, split, or hyperparameters, so val MAE should not degrade;
if the calibrator alpha shifts to accommodate the wider raw spread, the
existing `val_mae(alpha) <= max(val_mae(1.0), 220) + 5` budget still guards
sealed MAE, and the parent's 5.7 Elo of sealed-MAE headroom is preserved by
construction.

Realistic outcome: sealed `middle_decile_calibration` drops materially from
174.00 (target: <=130 conservative, <100 stretch); `test_mae` stays in the
240-250 band; `spearman` stays near 0.666; `overall_satisfied` becomes true if
the mid-decile gap crosses 100.
