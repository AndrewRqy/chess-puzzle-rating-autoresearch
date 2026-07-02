# Downloaded Papers

## 1. Predicting Chess Puzzle Difficulty with Transformers (GlickFormer)
- **File**: `2410.11078_glickformer_puzzle_difficulty.pdf`
- **Authors**: Szymon Miłosz, Paweł Kapusta (Lodz University of Technology)
- **Year**: 2024
- **arXiv**: https://arxiv.org/abs/2410.11078
- **Why relevant**: Directly addresses the research hypothesis. Trained on 4.2M Lichess puzzles, uses ChessFormer backbone with factorized transformer over move sequences. Reports MAE=217.71 (Factorized Encoder GlickFormer) vs. 227.00 (ChessFormer baseline). Placed 11th at IEEE BigData 2024 Cup.

## 2. Chess Rating Estimation from Moves and Clock Times Using a CNN-LSTM (RatingNet)
- **File**: `2409.11506_chess_rating_cnnlstm.pdf`
- **Authors**: Michael Omori, Prasad Tadepalli (Oregon State University)
- **Year**: 2024
- **arXiv**: https://arxiv.org/abs/2409.11506
- **Why relevant**: Applied to the 2024 IEEE Big Data Cup puzzle difficulty dataset. Uses CNN + BiLSTM (no hand-crafted features). Reports MAE=183 on game rating estimation; competitive results on puzzles. Code available at https://github.com/AstroBoy1/RatingNet.

## 3. Mastering Chess with a Transformer Model (ChessFormer)
- **File**: `2409.12272_mastering_chess_transformer.pdf`
- **Authors**: Daniel Monroe, LC Zero Team
- **Year**: 2024
- **arXiv**: https://arxiv.org/abs/2409.12272
- **Why relevant**: Introduces the ChessFormer architecture (Smolgen positional bias, learned attention over 64 board squares) used as backbone in GlickFormer. Reaches Elo 2347 and 93.5% puzzle-solving. Provides spatial representations useful for difficulty models.

## 4. Maia: Aligning Superhuman AI with Human Behavior (Chess) — original paper
- **File**: `2006.01855_maia_original.pdf`
- **Authors**: Reid McIlroy-Young, Siddhartha Sen, Jon Kleinberg, Ashton Anderson
- **Year**: 2020
- **arXiv**: https://arxiv.org/abs/2006.01855
- **Why relevant**: Foundational work on skill-aware chess models. Maia is a CNN trained on games from specific Lichess rating bands (1100–1900). Provides rating-conditioned move-probability signals used by top IEEE BigData Cup teams as features.

## 5. Maia-2: A Unified Model for Human-AI Alignment in Chess
- **File**: `2409.20553_maia2.pdf`
- **Authors**: Zhang, Jacob, et al. (CSSLab)
- **Year**: 2024
- **arXiv**: https://arxiv.org/abs/2409.20553
- **Why relevant**: Successor to Maia; unified skill-conditioned move prediction. Used by BigData Cup competitors combined with Stockfish/Leela features for difficulty prediction via gradient boosting.

## 6. Human-Aligned Chess with a Bit of Search
- **File**: `2410.03893_human_aligned_chess_search.pdf`
- **Authors**: Yiming Zhang, Athul Paul Jacob, et al.
- **Year**: 2024
- **arXiv**: https://arxiv.org/abs/2410.03893
- **Why relevant**: Explores combining human move prediction with small amounts of search, relevant to modeling "how a human of Elo X would solve this puzzle."

## 7. Pretraining Transformers for Chess Puzzle Difficulty Prediction
- **File**: `7603_pretraining_transformers_puzzle.pdf`
- **Authors**: Szymon Miłosz et al. (FedCSIS/Annals CSIS Vol. 43)
- **Year**: 2024
- **URL**: https://annals-csis.org/Volume_43/drp/pdf/7603.pdf
- **Why relevant**: Companion / extended version of GlickFormer approach; details pretraining strategies for chess puzzle difficulty estimation.

## Non-downloaded but noted (behind IEEE paywall / referenced only)
- Zysko et al., "IEEE Big Data Cup 2024 Report: Predicting Chess Puzzle Difficulty at KnowledgePit.ai" — competition report, IEEE Xplore doc 10825289.
- "Estimating Chess Puzzle Difficulty Without Past Game Records Using a Human Problem-Solving Inspired Neural Network Architecture" — IEEE Xplore doc 10826087 (2nd place BigData Cup 2024, auxiliary tasks: move prediction, theme prediction, position uncertainty).
- "The bread emoji Team's Submission…" — IEEE Xplore doc 10826037 (Maia-2 + hand-crafted + Stockfish/Leela features + gradient-boosted decision tree; 3rd place).
- "Pairwise Learning to Rank for Chess Puzzle Difficulty Prediction" — IEEE Xplore doc 10825356.
- Hristova, Guid, Bratko (2014) "Assessing the difficulty of chess tactical problems" — foundational cognitive-model work on puzzle difficulty.
