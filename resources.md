# Resources Catalog

## Summary

Resources gathered for **Predicting Lichess Chess-Puzzle Difficulty Ratings from Position**.

- **Papers**: 7 PDFs downloaded (all key modern approaches to chess puzzle / rating prediction).
- **Datasets**: 1 primary dataset — Lichess Puzzles (6M+ rows, 865 MB, 3 parquet shards).
- **Code repositories**: 3 cloned (RatingNet reference impl, Aimchess baseline predictor, Maia chess models).

## Papers

Total papers downloaded: 7. See `papers/README.md` for annotations.

| Title | Authors | Year | File | Key info |
|-------|---------|------|------|----------|
| Predicting Chess Puzzle Difficulty with Transformers (GlickFormer) | Miłosz, Kapusta | 2024 | `papers/2410.11078_glickformer_puzzle_difficulty.pdf` | SOTA MAE 217.71 on 4.2M puzzles; ChessFormer backbone + factorized temporal attention |
| Chess Rating Estimation from Moves and Clock Times (CNN-LSTM / RatingNet) | Omori, Tadepalli | 2024 | `papers/2409.11506_chess_rating_cnnlstm.pdf` | CNN + BiLSTM; MAE 182 on player rating; applied to BigData Cup puzzle set |
| Mastering Chess with a Transformer Model (ChessFormer) | Monroe, LC0 Team | 2024 | `papers/2409.12272_mastering_chess_transformer.pdf` | Smolgen positional bias; 2347 Elo; GlickFormer's backbone |
| Aligning Superhuman AI with Human Behavior (Maia) | McIlroy-Young et al. | 2020 | `papers/2006.01855_maia_original.pdf` | Skill-conditioned CNN move predictors; used for engine-derived features |
| Maia-2: Unified Model for Human-AI Alignment | Zhang, Jacob et al. | 2024 | `papers/2409.20553_maia2.pdf` | Successor to Maia; single skill-conditioned network |
| Human-Aligned Chess with a Bit of Search | Zhang, Jacob et al. | 2024 | `papers/2410.03893_human_aligned_chess_search.pdf` | Rating-conditioned move + small search |
| Pretraining Transformers for Chess Puzzle Difficulty Prediction | Miłosz et al. | 2024 | `papers/7603_pretraining_transformers_puzzle.pdf` | FedCSIS journal version, pretraining strategies |

## Datasets

Total downloaded: 1 primary (~865 MB local; git-ignored).

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Lichess Chess Puzzles | huggingface.co/datasets/Lichess/chess-puzzles | 865 MB, ~6M rows | Rating regression | `datasets/lichess_puzzles/data/train-*.parquet` | Verified: 2M rows per shard; Rating mean 1473, std 547; RatingDeviation median 79 |

See `datasets/README.md` for schema, download instructions, and modeling notes (feature leakage, mirroring, uncertainty-aware training).

## Code Repositories

Total cloned: 3.

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| RatingNet (CNN-BiLSTM) | github.com/AstroBoy1/RatingNet | Paper reference impl of chess rating estimator | `code/RatingNet/` | PyTorch; `src/chess_rating_net.py` has the model; weights on Google Drive |
| Aimchess Puzzle Rating Prediction | github.com/ieshuaganocry/aimchess-puzzle-rating-prediction | Shallow feature + sklearn baseline | `code/aimchess-puzzle-rating-prediction/` | Includes pickled model with `dill`; useful as low-bar baseline |
| Maia Chess | github.com/CSSLab/maia-chess | Skill-conditioned move prediction models | `code/maia-chess/` | Pre-trained Elo-bracket weights in `maia_weights/`; use for engine-derived features |

See `code/README.md` for how each fits into the research plan.

## Resource Gathering Notes

### Search strategy
- Started with the paper-finder service (unavailable — service not running); fell back to WebSearch on arXiv + Google Scholar.
- Focused on: (a) direct puzzle-rating prediction, (b) related chess-rating estimation (from games), (c) architectural backbones (ChessFormer), (d) skill-conditioned move predictors (Maia).
- Cross-referenced the IEEE BigData 2024 Cup for competitive baselines.

### Selection criteria
- **Directly on-task**: GlickFormer, RatingNet, Aimchess, IEEE BigData Cup entries — kept all accessible ones.
- **Architectural**: ChessFormer (backbone), Maia (skill-conditioned features).
- **Foundational cognitive-model work** (Bratko, Hristova, Guid) noted in `papers/README.md` but not downloaded — behind paywalls or older, not required for reproducing SOTA.

### Challenges encountered
- Paper-finder service not running at localhost:8000.
- IEEE BigData 2024 Cup submission papers (2nd, 3rd place; competition report; pairwise learning-to-rank paper) are behind IEEE Xplore paywall. Details recovered indirectly via GlickFormer's related-work section and search-result summaries.
- `wget` unavailable in environment; used `curl -sL` for all downloads.

### Gaps and workarounds
- IEEE-paywalled papers → captured methodology summaries in `literature_review.md` from search-result snippets and cross-references.
- RatingNet's pretrained weights on Google Drive not automatically retrieved. The experiment runner can either download them manually or retrain from scratch on the Lichess parquet.

## Recommendations for Experiment Design

Based on gathered resources:

1. **Primary dataset**: Lichess Puzzles (already downloaded, `datasets/lichess_puzzles/data/`). Recommend starting with a 500k–1M random subsample for iteration, then full dataset for final numbers.

2. **Baseline methods** (in order of implementation):
   - Global mean (~1516) — matches the "global-mean baseline" mentioned in the hypothesis; expected MAE ≈ 430.
   - Median-by-move-count (aimchess Method 2) — MAE ≈ 380.
   - Shallow tactical features + LightGBM/XGBoost regressor.
   - Small CNN over 16-channel piece-plane encoding.
   - CNN + BiLSTM (RatingNet-style, our reference impl in `code/RatingNet/`).
   - **Hybrid (target model per hypothesis)**: shallow features concatenated with CNN embedding → MLP head.

3. **Evaluation metrics**:
   - Primary: **MAE** on held-out test split (hypothesis threshold: < 250).
   - Secondary: MSE (for BigData Cup comparability), Spearman ρ, accuracy within 1/2/3 × RatingDeviation, MAE bucketed by number-of-solution-moves.

4. **Code to adapt / reuse**:
   - `code/RatingNet/src/chess_rating_net.py` for CNN encoder + BiLSTM.
   - `code/RatingNet/src/format_data.py` for PGN → tensor logic (needs adaptation to consume the puzzles parquet).
   - `code/aimchess-puzzle-rating-prediction/pre_rating_methods_comparison.py` for shallow-baseline patterns.
   - `python-chess` (already installed) for FEN parsing, legal-move generation, material counts.

5. **Key methodological notes** (copied from `literature_review.md`):
   - Exclude `NbPlays`, `RatingDeviation`, `Popularity` from inputs — they leak the target.
   - Mirror board when Black to move.
   - Uncertainty-aware training: sample target from `N(Rating, RatingDeviation²)` clipped to ±3 RD.
   - Watch for training collapse to the dataset mean; if it happens, drop LR to 1e-6, switch to RMSprop, and use cyclical restarts (GlickFormer's recipe).
