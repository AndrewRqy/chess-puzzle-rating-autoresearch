# Cloned Code Repositories

## 1. RatingNet (Omori & Tadepalli, 2024)
- **URL**: https://github.com/AstroBoy1/RatingNet
- **Location**: `code/RatingNet/`
- **Paper**: arXiv:2409.11506 — `papers/2409.11506_chess_rating_cnnlstm.pdf`
- **Purpose**: Reference implementation of a CNN + BiLSTM chess rating estimator. Also evaluated on the 2024 IEEE BigData Cup puzzle dataset.
- **Key files**:
  - `src/chess_rating_net.py` — CNN encoder + BiLSTM regression head; PyTorch. Ratings normalized by mean=1514, std=366.
  - `src/format_data.py` — PGN → training tensors pipeline.
  - `src/game_analysis.py` — inference / analysis script.
  - `requirements.txt` — PyTorch + `chess==1.10.0` + scikit-learn.
- **How to use for our research**:
  - The CNN architecture (positional feature extractor from FEN) is directly reusable.
  - BiLSTM over move sequence is a strong sequential baseline architecturally similar to what we should build.
  - Rating-normalization scheme is a good reference for our training target.
- **Blockers observed**: Trained model weights are on Google Drive (not automatically retrieved); training pipeline expects raw Lichess PGN files rather than the puzzles parquet — we will need to build a puzzle-specific data loader.

## 2. Aimchess Puzzle Rating Prediction
- **URL**: https://github.com/ieshuaganocry/aimchess-puzzle-rating-prediction
- **Location**: `code/aimchess-puzzle-rating-prediction/`
- **Purpose**: A production baseline from aimchess.com. Uses hand-crafted features + a pre-trained sklearn model (`model.pickle`).
- **Key files**:
  - `predictor.py` — `predict_rating(fen, solution)` loading a pickled model + feature extractor.
  - `pre_rating_methods_comparison.py` — Compares 3 methods:
    1. Constant baseline (1500).
    2. Move-count median-lookup baseline.
    3. Full pickled model prediction.
  - `model.pickle` — Sklearn model (loaded via `dill`) with `extract_regular_features` function embedded.
- **How to use for our research**:
  - Provides ready-made **shallow-feature baselines** (constant mean, median-by-move-count) that we can reproduce.
  - The pickled model demonstrates the feasibility of shallow-feature approaches; useful as a lower-bound comparison.
- **Note**: The `model.pickle` is included in the repo (LFS-less binary). Loading requires the `dill` package.

## 3. Maia Chess (CSSLab)
- **URL**: https://github.com/CSSLab/maia-chess
- **Location**: `code/maia-chess/`
- **Paper**: arXiv:2006.01855 — `papers/2006.01855_maia_original.pdf`
- **Purpose**: Skill-conditioned human move prediction models trained on Lichess games at Elo bands [1100, 1200, …, 1900]. Provides a way to derive a "solvability by rating X" signal for a puzzle position, which the 3rd-place BigData Cup team used as features for their difficulty model.
- **Key files**:
  - `maia_weights/` — Pre-trained network weights for each Elo bracket.
  - `move_prediction/` — Inference + evaluation scripts.
  - `maia_chess_backend/` — Backend built on Leela Chess Zero.
- **How to use for our research**:
  - **Feature engineering (optional advanced experiment)**: For a puzzle position + best move, query multiple Maia models. Puzzles where low-rated Maias don't find the best move but high-rated ones do are (empirically) harder. Convert to a numeric feature per Elo band.
  - Difficulty can be recovered as the Elo bracket at which prediction switches to the puzzle's expected best move.

## Related repositories NOT cloned (noted for reference)
- **Leela Chess Zero** — https://github.com/LeelaChessZero/lc0 (heavy C++ engine; only needed if we recreate ChessFormer end-to-end).
- **Lichess database** — https://database.lichess.org/ (raw PGN dumps; alternate source for the puzzle CSV).
- **Lichess papers list** — https://github.com/lichess-org/papers (bibliography of published Lichess research).
