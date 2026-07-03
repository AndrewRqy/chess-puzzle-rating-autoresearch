# Rule Maker Rationale

Task: **Predicting Lichess Chess-Puzzle Difficulty Ratings from Position.**
The runner produces a `predict(rows: list[dict]) -> list[float]` function
that takes puzzle rows (FEN + Moves + Themes) and returns predicted
Glicko-2 Elo ratings.

## Properties chosen

- **test_mae** (direction: min). The primary regression metric named in
  the idea's hypothesis and in the `expected_outputs[metrics].fields`
  list. Interpretable in Elo points. Directly comparable to published
  numbers (GlickFormer SOTA MAE ≈ 217.71, ChessFormer baseline 227.00,
  RatingNet ~200–220, global-mean baseline ~430).

- **spearman** (direction: max). Rank correlation captures the model's
  ability to *order* puzzles by difficulty even when its absolute
  calibration is off. A predictor that biases every prediction by a
  constant would still score well here — a useful cross-check against
  the MAE metric, which is sensitive to bias. Named in the idea's
  `expected_outputs[metrics].fields` and evaluation criteria.

- **middle_decile_calibration** (direction: min). Largest absolute gap
  between mean predicted rating and mean actual rating inside any
  100-Elo bin whose bin center falls in [1400, 2000]. Bins the puzzles
  by their true rating (unknown to the runner), so a mean-only
  predictor cannot game it. Directly maps to the idea's evaluation
  criterion "Calibration error < 100 Elo on middle deciles (rating
  1400-2000)".

## Properties considered and rejected

- **rmse**: named in the idea's fields but redundant with MAE for a
  scalar success rule; RMSE is more sensitive to outliers but conveys
  no independent signal for pass/fail judgement in this task.
  Rejected to keep the property set small (1–3 is ideal per guidelines).

- **reproducibility_across_seeds** (delta MAE < 20 across two random
  seeds): named in evaluation_criteria but not feasibly measured from a
  single runner artifact. It would require the runner to re-train under
  two seeds and expose both models. Given the 6-hour compute budget and
  the fact that re-training doubles it, we defer this to a
  supervisor-driven follow-up. Documented here so the user can add a
  second-seed protocol later if they want to enforce it.

- **accuracy_within_1_2_3_RD**: mentioned in the literature review and
  the GlickFormer paper. Rejected because it requires access to each
  puzzle's `RatingDeviation` at score time (available, but the metric
  duplicates MAE for our threshold analysis and adds complexity).

## Targets

- **test_mae = 250.0** Elo. Source: the idea's hypothesis states an
  explicit "MAE < 250 Elo" target. The literature review confirms this
  is a realistic threshold: any real learned model beats it, so it is a
  meaningful minimum-competence bar without being a stretch.
  The global-mean baseline is empirically ~430 in this data, so passing
  250 implicitly satisfies the "strictly better than global-mean"
  criterion too.

- **spearman = 0.4**. Source: the idea's evaluation_criteria explicitly
  requires Spearman ρ > 0.4 on the held-out split.

- **middle_decile_calibration = 100.0** Elo. Source: the idea's
  evaluation_criteria requires "Calibration error < 100 Elo on middle
  deciles (rating 1400-2000)".

`success_rule` is `ALL_PROPERTIES_SATISFIED` (default). Missing any
one of {accurate, well-ranked, well-calibrated} would render the
research contribution weak, so a conjunction is the right rule.

## Baseline

- **Source**: no separate baseline file is invoked by `scoring/eval.py`.
  The `test_mae` target of 250 Elo is already well below the global-mean
  baseline (~430 Elo), so beating the baseline is implicit.
- **Why this is fair**: absolute-target properties are appropriate when
  a reasonable numeric benchmark is available in the literature (here,
  the idea's own hypothesis + GlickFormer's published MAE 217.71 give
  us both a floor and a ceiling). A separate baseline module would
  introduce an unnecessary dependency; the user can add one later if
  they want a relative ratio metric.
- Reference baselines documented in the literature review for reference:
  - Global mean ≈ 1516 → MAE ≈ 430.
  - Median-by-move-count → MAE ≈ 380.
  - Shallow-feature GBM → MAE ≈ 260–290.
  - CNN + BiLSTM (RatingNet-family) → MAE ≈ 200–220.
  - GlickFormer (SOTA) → MAE 217.71.

## Metric robustness

Threats to the scoring rule and how eval.py guards against each:

- **Constant predictor** (returns 1516 for every puzzle): fails all
  three metrics simultaneously. MAE will be ≈ 430 (fails target 250),
  Spearman ρ will be 0.0 (fails target 0.4), and the middle-decile
  calibration gap will be > 100 Elo for at least one bin whose true
  mean rating is far from 1516 (fails target 100).

- **Median predictor** or any fixed-scalar predictor: same story;
  Spearman = 0 kills it.

- **Rating leak via unauthorized columns**: `scoring/eval.py` strips
  every key beginning with `_` from the row before passing to the
  runner. The true rating lives on the `_Rating` key and never crosses
  that boundary.

- **Test-set reconstruction from `interface.md`**: the hash constant
  and modulus that select the sealed slice live only in
  `scoring/eval.py`. `interface.md` shows the runner an *unrelated*
  example hash for their own validation split.

- **Random-jitter predictor**: a predictor that adds noise to a fixed
  mean will still score MAE ≈ 430 and near-zero Spearman.

- **Overfitting to the calibration bin structure**: the bin edges (100
  Elo, [1400, 2000]) are not in `interface.md`; a runner cannot
  regress specifically to satisfy this metric without also improving
  its overall MAE.

- **Empty middle-Elo bin**: guarded by a minimum sample count (20 per
  bin) before including a bin in the calibration gap. With a 2000-row
  test slice on Lichess puzzles, the 1400–1999 range typically
  contributes 500+ puzzles, so all six 100-Elo bins are covered.

## Sealed test split

- Read from local parquet shards under
  `datasets/lichess_puzzles/data/train-*.parquet`.
- Filter: `RatingDeviation < 80` (matches the idea's subsampling rule).
- Hash-based holdout: a puzzle is in the test slice iff
  `md5(PuzzleId) % 100 == 37`. That constant is private to `eval.py`.
- Cap: 2000 puzzles, deterministic sub-sample using a fixed seed for
  scoring runtime bounds.
- Deterministic across runs: the same puzzles are chosen every time,
  so scoring is reproducible.

## Known limitations

- **Runner has read access to the full parquet.** A malicious runner
  could enumerate hash values and identify the held-out puzzles. In an
  adversarial evaluation this test set would need to be genuinely
  hidden (files under `data/.test/` with tight permissions). For a
  cooperative research pipeline this is acceptable, and the runner is
  told in `interface.md` not to attempt reconstruction.

- **Reproducibility-across-seeds criterion is not enforced.** See
  the "Properties considered and rejected" section. The user can add
  a two-seed protocol by (a) requiring `predict(rows, seed=int)` in
  a v2 interface and (b) invoking it twice in eval.py.

- **No secondary metrics logged.** The results.json includes only the
  three pass/fail properties. RMSE, per-bucket MAE, per-theme MAE, and
  accuracy-within-N-RD are useful diagnostics but do not affect the
  success rule; the runner can compute them itself and stash them in
  its own report.

- **Absolute — not baseline-relative — targets.** If the Lichess
  rating distribution shifts substantially (e.g., a re-rating event on
  the puzzle side), the numeric targets may drift. Re-anchor by
  re-reading the current global-mean baseline before large gaps in
  time.
