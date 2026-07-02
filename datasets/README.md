# Datasets

Large data files are NOT committed to git (see `.gitignore`). Follow the download instructions to reproduce.

## Lichess Chess Puzzles (Primary Dataset)

- **Source**: HuggingFace `Lichess/chess-puzzles`
- **Local location**: `datasets/lichess_puzzles/data/train-*.parquet`
- **Size on disk**: ~865 MB across 3 parquet shards
- **Rows**: ~6,014,381 total puzzles (this download: 3 shards, ≈2M rows each verified via first shard)
- **License**: CC0 1.0 (public domain, per Lichess database policy)
- **Format**: Apache Parquet (fast columnar)

### Schema (verified from downloaded parquet)

| Column | Type | Description |
|--------|------|-------------|
| PuzzleId | string | Unique ID (URL: `https://lichess.org/training/{PuzzleId}`) |
| GameId | string | Source Lichess game |
| FEN | string | Board position BEFORE the opponent's setup move; puzzle "starts" AFTER first move in `Moves` |
| Moves | string | Space-separated UCI moves (first move is opponent's setup, subsequent are solution) |
| Rating | uint16 | Glicko-2 puzzle difficulty (**TARGET**) |
| RatingDeviation | uint16 | Glicko-2 rating deviation (uncertainty of target) |
| Popularity | int8 | 100 × (up − down) / (up + down), range [-100, 100] |
| NbPlays | uint32 | Number of times solved |
| Themes | list[str] | Tags: mate, fork, pin, middlegame, endgame, short, long, crushing, etc. |
| OpeningTags | list[str] or None | Opening name (only if puzzle appears before move 20) |

### Verified Statistics (from first shard, N=2,004,794)
- **Rating**: mean = 1473, std = 547, min = 399, median = 1419, max = 3254
- **RatingDeviation**: mean = 87, median = 79 (majority in [75, 90] band — high confidence)
- Distribution consistent with GlickFormer paper (4.2M subset: mean 1516, std 543)

### Download Instructions

Preferred method (used to produce the local copy):

```python
from huggingface_hub import hf_hub_download
import os
os.makedirs('datasets/lichess_puzzles', exist_ok=True)
for i in range(3):
    fname = f'data/train-{i:05d}-of-00003.parquet'
    hf_hub_download(
        repo_id='Lichess/chess-puzzles',
        filename=fname,
        repo_type='dataset',
        local_dir='datasets/lichess_puzzles',
    )
```

Alternative (loads into HF Datasets object; will download & cache):

```python
from datasets import load_dataset
ds = load_dataset("Lichess/chess-puzzles", split="train")
```

Direct CSV download (updated daily, ~350 MB compressed):

```bash
mkdir -p datasets/lichess_puzzles_csv
curl -L -o datasets/lichess_puzzles_csv/puzzles.csv.zst \
    https://database.lichess.org/lichess_db_puzzle.csv.zst
zstd -d datasets/lichess_puzzles_csv/puzzles.csv.zst
```

### Loading

```python
import pandas as pd
import glob

# Load all shards
files = sorted(glob.glob("datasets/lichess_puzzles/data/train-*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

# Or load single shard (faster iteration)
df = pd.read_parquet("datasets/lichess_puzzles/data/train-00000-of-00003.parquet")
```

### Sample Data
See `datasets/lichess_puzzles/samples.json` for 10 example puzzles.

### Notes for Modeling
- **Target**: `Rating` (int, ~[400, 3300]). The GlickFormer paper suggests standardizing by `(Rating - 1516) / 543`.
- **Uncertainty-aware training**: Sample target from `N(Rating, RatingDeviation^2)` and clip to ±3 RD (per GlickFormer). This regularizes and encodes noise in Glicko-2 labels.
- **Puzzle position parsing**: Apply the FIRST move in `Moves` to the FEN — this produces the position where the solver is on move. `Moves[1:]` is the solution sequence.
- **Perspective mirroring**: To standardize input, mirror the board when it's Black to move (all reviewed papers do this).
- **Themes** are useful auxiliary labels (mate patterns, tactical motifs); can be used for multi-task learning as in the 2nd-place BigData Cup submission.
- **NbPlays / Popularity / RatingDeviation** are metadata leaks — do NOT use as inputs (they encode the target). Only FEN + Moves are legitimate features; Themes are borderline (may be predicted at inference or held out).
- **Suggested split**: random 90/5/5 train/val/test on `PuzzleId`. Deduplicate on `GameId` if fair generalization is needed. GlickFormer used 4,158,000 train / 42,000 test (~99/1).
