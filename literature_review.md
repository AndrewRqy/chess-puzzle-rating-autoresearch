# Literature Review — Predicting Lichess Puzzle Difficulty from Position

## Research area overview

The task is regression: given a chess puzzle (FEN + best-move sequence), predict its empirical Glicko-2 difficulty on Lichess. Difficulty ratings are earned exactly like player ratings — each puzzle attempt is an "opponent match" (win = solve, loss = fail), and its rating drifts to whatever population of players consistently solves it. This makes the ratings noisy but well-calibrated at scale.

The field is small but grew rapidly in 2024 because of the **IEEE BigData 2024 Cup: Predicting Chess Puzzle Difficulty** at KnowledgePit.ai (metric: MSE on Elo). All modern approaches fall into three families:

1. **Transformer over positions/moves** — GlickFormer (Miłosz & Kapusta), ChessFormer baseline; capture spatial (piece config) + temporal (move sequence) structure.
2. **CNN + BiLSTM** — RatingNet (Omori & Tadepalli); no hand-crafted features, sequence-aware.
3. **Feature engineering + tree ensemble** — Hand-crafted tactical features (material, mobility, king safety), plus signals from Maia, Leela, Stockfish, plumbed into gradient-boosted decision trees (LightGBM). This was the 3rd-place BigData Cup submission.

The hybrid approach in the research hypothesis (shallow tactical features + learned representations) sits between families 2 and 3.

## Key papers

### Paper 1 — GlickFormer (Miłosz & Kapusta, arXiv:2410.11078, 2024)
- **Contribution**: Transformer for puzzle-difficulty regression. Uses ChessFormer as spatial backbone, factorized temporal attention (à la ViViT) over the sequence of positions in the puzzle. Two variants: Factorized Encoder (better) and Factorized Self-Attention.
- **Encoding**: Each position → 16-channel 8×8 binary tensor (12 piece-presence planes + 4 move-indicator planes for previous / next move). Board mirrored so mover is always "white". Positions sampled every other ply.
- **Loss**: MSE on standardized Glicko-2 rating. **Uncertainty augmentation**: sample training targets from `N(rating, RatingDeviation²)` clipped to ±3 RD — natural regularizer.
- **Dataset**: 4.2M Lichess puzzles; 4,158,000 train / 42,000 test (~99/1 split).
- **Results** (test set):

  | Model | MAE | MAZ | Acc<1RD | Acc<2RD | Acc<3RD |
  |-------|-----|-----|---------|---------|---------|
  | ChessFormer baseline | 227.00 | 2.68 | 25.42% | 48.21% | 65.29% |
  | Fact. Self-Attention GlickFormer | 221.80 | 2.62 | 26.13% | 48.54% | 66.08% |
  | Fact. Encoder GlickFormer | **217.71** | **2.57** | **27.00%** | **50.45%** | **67.66%** |

- **Competition score**: preliminary MSE 75,995 → final 158,292 on the BigData Cup test set (placed 11th).
- **Relevance**: Sets the modern SOTA for the task. Gives us the target to beat and a strong architecture template.

### Paper 2 — RatingNet / CNN-BiLSTM (Omori & Tadepalli, arXiv:2409.11506, 2024)
- **Contribution**: 4-layer CNN over 8×8×N piece planes → BiLSTM over move sequence → per-move rating prediction. First neural rating estimator with no hand-crafted features.
- **Ablation finding**: clock time (irrelevant for puzzles) contributes ~5% of MAE reduction; the CNN+LSTM alone is already strong.
- **Result**: MAE 182 for player-rating estimation from games (1.2M game benchmark); "competitive" (unquantified in text) on the BigData Cup puzzle set.
- **Relevance**: The reference implementation is public (`code/RatingNet/`). Their CNN-BiLSTM is a great "learned representation" component to combine with shallow tactical features per our hypothesis.

### Paper 3 — ChessFormer / Mastering Chess with a Transformer (Monroe & LC Zero, arXiv:2409.12272, 2024)
- **Contribution**: Chess-specific transformer with **Smolgen** — a learnable per-square attention-bias mechanism that reflects chess movement patterns rather than Euclidean distance. Reaches Elo 2347 and 93.5% puzzle-solving on Lichess puzzles as a **playing** engine.
- **Relevance**: Backbone used inside GlickFormer. If we go transformer, we should reuse this architecture / its ideas rather than start from scratch. Also demonstrates that a strong "chess understanding" transformer implicitly encodes tactical difficulty.

### Paper 4 — Maia and Maia-2 (McIlroy-Young et al.; Jacob et al., arXiv:2006.01855 / arXiv:2409.20553)
- **Contribution**: Skill-conditioned move prediction. Maia is a family of CNNs trained on Lichess games from specific Elo bands (1100–1900). Maia-2 unifies these into a single skill-conditioned model.
- **Difficulty-prediction application**: Query Maia at each Elo bracket on the puzzle FEN; the lowest bracket at which Maia's top move matches the puzzle's best move correlates strongly with puzzle rating (used by 3rd-place BigData Cup team as GBDT features).
- **Relevance**: Provides "engine-derived rating-conditioned signals" for the hybrid approach.

### Paper 5 — Human-Aligned Chess with a Bit of Search (Zhang, Jacob et al., arXiv:2410.03893)
- **Contribution**: Combines rating-conditioned move prediction with lightweight search. Shows that adding a small number of Stockfish rollouts to Maia's predictions gives a strong human-like agent.
- **Relevance**: Motivates using engine features (best-move quality, evaluation drop, PV depth) as difficulty proxies.

### Paper 6 — Pretraining Transformers for Chess Puzzle Difficulty Prediction (Miłosz et al., FedCSIS 2024)
- **Contribution**: Companion / journal version of GlickFormer. Explores pretraining objectives for chess-puzzle transformers.
- **Relevance**: Additional design details for the GlickFormer approach.

### IEEE BigData 2024 Cup (top submissions, cited but not downloadable — behind IEEE paywall)
- **1st place** (details limited): approach summary not in accessible papers.
- **2nd place** (IEEE 10826087): NN inspired by human problem-solving. Multi-task: predicts (a) best move (auxiliary), (b) themes (auxiliary), (c) position uncertainty → difficulty. Trained without past game records.
- **3rd place, "bread emoji" team** (IEEE 10826037): Fine-tuned Maia-2 + hand-crafted features + Stockfish/Leela features → **LightGBM**. This is closest to our hypothesis.
- **Pairwise Learning to Rank** (IEEE 10825356): Frames the problem as ranking rather than regression.
- **Competition report** (IEEE 10825289): Overview and metric = MSE of un-normalized Elo (399–3331).

## Common methodologies

- **Position encoding**: 8×8 grid with per-piece channels (12 planes) + move planes (2 for previous move, 2 for anticipated next move). Board mirrored to solver's perspective. Universal across all reviewed papers.
- **Move sequence**: Encode each successive position after each move in the solution; feed to a temporal model (transformer or LSTM).
- **Target normalization**: Standardize by mean 1516, std 543 (from GlickFormer). RatingDeviation-based label noise is a widely useful regularizer.
- **Optimizer**: GlickFormer required RMSprop with very small LR (1e-6) and cyclical restarts to avoid mode collapse (predicting the dataset mean). Higher LR / Adam converged to the mean. This is a nontrivial practical finding to remember.

## Standard baselines (in order of increasing sophistication)

| Baseline | MAE (approx) | Notes |
|----------|--------------|-------|
| Constant mean (~1516) | ≈**430** | Trivial; equals dataset std × ≈0.79. Must beat. |
| Median rating by number of solution moves | ~380 | Aimchess "Method 2". Very cheap; captures gross difficulty trend. |
| Handcrafted features + regressor (Aimchess) | Unknown (proprietary) | Shallow features + sklearn. |
| CNN-BiLSTM (RatingNet, Omori & Tadepalli) | ~200–220 est. on puzzles | Strong learned baseline. |
| ChessFormer (Monroe) as regressor | 227.00 | GlickFormer paper baseline. |
| **GlickFormer (Factorized Encoder)** | **217.71** | Current SOTA in the peer-reviewed literature. |
| GBDT on hand + engine features (3rd place BigData Cup) | ~205–215 est. | Different architecture family. |

Target from the research hypothesis: MAE < 250. Easily surpassable with any real learned model; realistic stretch target is MAE < 220 to match SOTA.

## Evaluation metrics

- **MAE** (Mean Absolute Error) — primary metric; in Elo points. Interpretable.
- **MSE** — used by the IEEE BigData Cup leaderboard (on un-normalized Elo).
- **MAZ** (Mean Absolute Z-Score) — `|r − r̂| / RD` — accounts for label uncertainty.
- **Accuracy within {1,2,3}×RD** — proportion of predictions inside 1 / 2 / 3 rating deviations of the true rating.
- **Bucketed MAE** — break down by number of solution moves (single-move puzzles are noisiest; 3–4 move puzzles are the sweet spot for learned models).
- **Spearman ρ / Pearson r** — useful sanity check.

## Datasets in the literature

- **Lichess Puzzles Database** (huggingface.co/datasets/Lichess/chess-puzzles). 6M+ puzzles; the de facto standard. All papers use a subset.
- **IEEE BigData 2024 Cup dataset** — held-out subset of Lichess puzzles used for the competition. Not clear if train/test splits are publicly available; may be gated at knowledgepit.ai.
- **Lichess games PGN** (database.lichess.org) — used for training Maia and RatingNet's game-rating estimator; not directly required for puzzle rating.

## Gaps and opportunities

- **Shallow interpretable features are underexplored.** Only Aimchess and the 3rd-place BigData Cup team use handcrafted tactical features; both combine them with black-box components. There's no clear ablation isolating the contribution of e.g. material balance, forcing sequences, or king exposure to explain what a neural network already learns.
- **Multi-task learning with themes** has been tried only in the 2nd-place BigData Cup work; the Themes column (fork, pin, mateInN, etc.) is a rich auxiliary target.
- **Cognitive-plausibility signals** — number of candidate moves at each ply (branching factor), Stockfish evaluation drop after the puzzle's setup move, PV entropy — are cheap and cognitively motivated.
- **Uncertainty-aware evaluation**: papers report MAE but rarely separately for high-RD (unreliable label) vs. low-RD puzzles.

## Recommendations for our experiment

**Design target**: Beat the MAE < 250 hypothesis threshold and aim toward MAE ≈ 220 (matching GlickFormer / RatingNet band).

### Recommended datasets
- **Primary**: full 6M-row Lichess puzzles parquet (already downloaded). Random 90/5/5 or 98/1/1 split by `PuzzleId`.
- **Subsampling for iteration**: 500k–1M puzzles is enough to develop and compare architectures; move to full-scale only for final numbers.

### Recommended baselines to implement (in order)
1. **Global mean predictor** (~1516) — hypothesis's baseline; MAE ≈ 430.
2. **Move-count median lookup** (aimchess Method 2) — MAE ≈ 380.
3. **Themes-only linear regression** (using Themes as bag-of-words features).
4. **Shallow-feature GBM (LightGBM/XGBoost)**: material balance, mobility, king safety, mating pattern flags, solution length, forced-move count, Stockfish eval delta if feasible. Target MAE 260–290.
5. **Small CNN over piece-plane encoding** (à la Maia but for regression). Target MAE 240.
6. **CNN + BiLSTM** (RatingNet-style). Target MAE 200–220.
7. **Hybrid**: concatenate shallow features + CNN embedding, then MLP. This is the direct implementation of the research hypothesis.

### Recommended evaluation metrics
- Primary: **MAE** on held-out test set.
- Secondary: **MSE** (matches BigData Cup), **Spearman ρ**, **Accuracy within 1/2/3 RD**, and **MAE bucketed by number of moves**.

### Methodological considerations
- **Feature/label leakage warning**: `NbPlays`, `RatingDeviation`, `Popularity` are computed after the puzzle is played and encode the target. Exclude from inputs.
- **Perspective normalization**: mirror board when Black to move (do this once at preprocessing).
- **Label noise**: sample target from `N(Rating, RatingDeviation²)` during training as a regularizer (GlickFormer's trick).
- **Optimization**: expect training instability with momentum optimizers; if the model collapses to the dataset mean, try RMSprop with small LR (1e-6) and cyclical restart per GlickFormer.
- **Compute budget**: full 4M-puzzle transformer training took multiple RTX 6000 GPUs. Our experiments will need to work at smaller scale; the CNN+MLP hybrid is the pragmatic sweet spot.
- **Themes as auxiliary**: multi-task loss with `mate / fork / pin / mateInN` prediction can improve rating prediction; also serves as label smoothing when Themes correlate with difficulty.
