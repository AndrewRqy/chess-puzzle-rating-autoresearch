# Artifact Protocol

The experiment_runner must produce the artifacts below. Conform exactly to
paths, signatures, and schemas. Any deviation will cause the post-run
scorer to fail.

## Files to produce

| Path | Purpose | Required |
|---|---|---|
| `src/predict.py` | Module exposing the callable being measured | yes |

## Entry points

### `src/predict.py`

Must define a top-level function:

```python
def predict(rows: list[dict]) -> list[float]:
    """Return a predicted Lichess Elo rating for each puzzle row."""
```

- `rows`: a list of dicts. Each dict has at minimum these keys:
  - `PuzzleId` (str): unique Lichess puzzle identifier
  - `FEN` (str): standard FEN string. The puzzle *starts* AFTER the
    first move in `Moves` is applied.
  - `Moves` (str): space-separated UCI moves. The first move is the
    opponent's setup move; subsequent moves form the solution sequence.
  - `Themes` (list[str] or None): may be empty. Do not treat as a
    required feature; production inputs may omit or empty this field.
- Returns: `list[float]` with the same length and order as `rows`.
  Each element is a real-valued predicted rating (typical range
  400–3300). No NaNs, no infinities.
- No side effects during the call: no network, no writes to disk
  outside a `.cache/` directory the runner may create for model
  weights.
- Deterministic: two invocations with identical `rows` (order
  preserved) must return element-wise identical outputs.
- Any model weights, feature extractors, or lookup tables the runner
  needs must be loaded at import time or lazily on first call and
  shipped inside the `src/` tree (or a `models/` sibling directory).

## Invocation

The post-run scorer will import `src/predict.py` via `importlib.util`,
call `predict(rows)` on a sealed held-out subset of the Lichess puzzle
dataset, and compare returned floats to the true ratings. The scorer
does NOT pass any arguments beyond `rows`; do not read `sys.argv` or
environment variables at inference time.

## Training data guidance

The training corpus is the local Lichess puzzles dataset at
`datasets/lichess_puzzles/data/train-*.parquet`
(~6M rows across 3 parquet shards). A sample of the schema is at
`datasets/lichess_puzzles/samples.json`. See `datasets/README.md` for
column definitions and modeling notes.

Feature/label leakage: the columns `NbPlays`, `RatingDeviation`, and
`Popularity` are computed from user-attempt logs AFTER the puzzle is
played and encode the target. Do NOT use them as input features.
Legitimate inputs are `FEN`, `Moves`, `Themes` (optional), and
`OpeningTags` (optional).

Held-out safety: to avoid accidental test-set overlap, exclude any
puzzle whose `PuzzleId` matches a hash-based holdout from training.
The exact holdout rule is sealed, but you can use the following
safe pattern to reserve a similarly-sized slice from your OWN data
for internal validation without touching the sealed test set:

```python
import hashlib
def is_my_val(puzzle_id: str) -> bool:
    return int(hashlib.md5(puzzle_id.encode()).hexdigest(), 16) % 100 < 5
```

The scorer uses a DIFFERENT constant for the sealed test slice. Do not
attempt to enumerate holdout candidates.

## What is measured (context only — do not tune)

- **test_mae** (direction: min, mean absolute error in Elo points on the
  sealed test slice).
- **spearman** (direction: max, Spearman rank correlation between
  predicted and true ratings).
- **middle_decile_calibration** (direction: min, largest absolute gap
  between mean predicted rating and mean actual rating across the
  middle Elo bins).

Test inputs, the baseline, and numeric targets are sealed. Do not read
`scoring/eval.py`, `scoring/targets.json`, or anything under `data/.test/`.
