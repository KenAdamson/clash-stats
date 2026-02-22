# ADR-003: Game State Embedding Model

**Status:** Proposed
**Date:** 2026-02-22
**Depends on:** ADR-001 (Feature Engineering), ADR-007 (Training Data Pipeline)

## Context

A game of Clash Royale is a trajectory through a high-dimensional state space. Two games can look superficially different (different cards, different positions, different timing) but be structurally identical (both are "opponent overcommits left lane, player punishes with opposite-lane push, snowballs into three-crown"). Current analysis treats each game as an isolated data point. Embeddings allow us to discover the latent structure — the manifold of game trajectories that separates wins from losses, aggression from defense, optimal play from misplay.

## Decision

### 1. Architecture: Temporal Convolutional Network (TCN)

The game event sequence is an ordered time series of variable length (20-100+ events per game). We need a model that:
- Handles variable-length sequences efficiently
- Captures both local patterns (card-response pairs) and global patterns (game-level tempo)
- Produces a fixed-size embedding regardless of game length
- Trains efficiently on hundreds to low thousands of examples

**Why TCN over alternatives:**

| Architecture | Pros | Cons | Verdict |
|-------------|------|------|---------|
| LSTM/GRU | Good at sequential dependencies | Slow training, vanishing gradients on long sequences, sequential inference | Viable but slower |
| Transformer | Attention captures long-range dependencies | Needs 10K+ examples, O(n²) attention, overkill for sequences of 50-100 | Future upgrade path |
| 1D-CNN | Fast, parallelizable, good local pattern detection | Limited receptive field without stacking | Too shallow alone |
| **TCN** | **Dilated convolutions = exponential receptive field, parallelizable, trains well on small data** | Less flexible than attention | **Best fit for current scale** |

The TCN uses dilated causal convolutions with exponentially increasing dilation factors (1, 2, 4, 8, 16, ...) so that with 6-7 layers, the receptive field covers the entire game sequence. Each layer sees a wider temporal context without losing resolution on local card interactions.

### 2. Model Structure

```
Input: (batch, seq_len, event_feature_dim)    # From ADR-001, dim ≈ 34
  │
  ▼
Card Embedding Layer (shared with other models)
  │
  ▼
TCN Encoder (6 blocks, channels: 64 → 64 → 128 → 128 → 256 → 256)
  │ Each block: dilated_conv → batch_norm → ReLU → dropout → dilated_conv → batch_norm → ReLU → dropout → residual
  │ Dilation factors: 1, 2, 4, 8, 16, 32
  │
  ▼
Global pooling: concatenate [mean_pool, max_pool, last_hidden] → dim 768
  │
  ▼
Projection head: Linear(768 → 256) → ReLU → Linear(256 → 128)
  │
  ▼
Game Embedding: 128-dimensional vector
  │
  ├──► Classification head: Linear(128 → 1) → Sigmoid    [for supervised training]
  └──► Contrastive projection: Linear(128 → 64)           [for contrastive training]
```

**Parameter count:** ~2M parameters. Trainable on CPU in minutes for datasets under 10K games.

### 3. Training Objectives

Three training phases, applied sequentially or jointly:

#### Phase 1: Supervised Win/Loss Classification
- Binary cross-entropy: predict win (1) vs loss (0)
- This is the simplest signal and gives the model a reason to learn game-relevant features
- Minimum data: 200 games (100W/100L) with replay data
- Expected performance at 200 games: 65-75% accuracy (above baseline of deck matchup alone)
- Expected performance at 2,000 games: 80-85%

#### Phase 2: Contrastive Learning (SimCLR-style)
- Positive pairs: two different games with the same outcome against the same archetype
- Negative pairs: win vs loss against the same archetype
- NT-Xent loss pulls same-outcome games together in embedding space and pushes different-outcome games apart
- This learns structure beyond just win/loss — it discovers *how* wins differ from losses
- Minimum data: 500 games (need enough same-archetype pairs)

#### Phase 3: Multi-Task Auxiliary Objectives
- Crown count regression: predict final crown differential (-3 to +3)
- Game duration prediction: predict whether game goes to overtime
- Elixir efficiency regression: predict final elixir differential
- These auxiliary signals force the embedding to capture richer game state information
- Minimum data: 500+ games

### 4. Embedding Applications

Once trained, the 128-dim embedding enables:

#### Game Similarity Search
```sql
-- "Show me games most similar to this loss"
-- Compute embedding for target game, find k-nearest neighbors by cosine similarity
SELECT battle_id, result, opponent_archetype,
       cosine_similarity(embedding, target_embedding) as similarity
FROM game_embeddings
ORDER BY similarity DESC
LIMIT 10;
```

This answers: "I just lost a weird game — have I seen this pattern before? What happened those times?"

#### Win/Loss Cluster Analysis
- Run UMAP or t-SNE on the embedding space
- Visualize: do wins and losses form distinct clusters? Are there sub-clusters within losses that correspond to different failure modes?
- Identify "close losses" (embeddings near the win cluster) vs "blowout losses" (far from any win pattern)

#### Archetype Fingerprinting
- Average the embeddings of all games against a given archetype
- The resulting "archetype centroid" captures the typical game flow against that deck type
- Distance between archetype centroids reveals which matchups play similarly

#### Anomaly Detection
- Games whose embeddings are far from any cluster centroid are "weird" games — unusual strategies, misplays, or novel opponent approaches
- Surface these for manual replay review

### 5. Embedding Storage

```sql
CREATE TABLE game_embeddings (
    battle_id TEXT PRIMARY KEY REFERENCES battles(battle_id),
    embedding BLOB NOT NULL,        -- 128 x float32 = 512 bytes
    model_version TEXT NOT NULL,     -- Track which model produced this
    win_probability REAL,            -- P(win) from classification head
    cluster_id INTEGER,              -- Assigned cluster from k-means/HDBSCAN
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_game_embeddings_cluster ON game_embeddings(cluster_id);
CREATE INDEX idx_game_embeddings_model ON game_embeddings(model_version);
```

Embeddings are recomputed when the model is retrained. The `model_version` column tracks provenance.

### 6. Manifold Geometry

The 128-dim embedding space will not be uniformly occupied — game trajectories live on a lower-dimensional manifold within it. Understanding the manifold geometry is where Ken's experience with high-dimensional manifold generation becomes directly applicable:

- **Intrinsic dimensionality estimation:** Use methods like MLE (Levina-Bickel) or TwoNN to estimate the true dimensionality of the game manifold. If 128-dim embeddings live on a 12-dim manifold, that tells us there are ~12 independent "factors of variation" in how games unfold.
- **Manifold interpolation:** Given two game embeddings (e.g., a clean win and a close loss against the same archetype), interpolate along the geodesic on the manifold. The intermediate points represent hypothetical games that "bridge" the two outcomes. If a decoder exists (ADR-006), these can be decoded into synthetic event sequences.
- **Curvature analysis:** Regions of high curvature on the manifold correspond to decision boundaries — small changes in play lead to large changes in outcome. These are the critical moments in a game. Identifying high-curvature regions tells you *where* the game is most sensitive to player decisions.
- **Density estimation:** Fit a normalizing flow or GMM to the embedding distribution. Low-density regions represent rare game types. If a particular matchup pushes games into low-density regions, that matchup is genuinely unusual and can't be modeled by interpolation from common patterns.

### 7. Implementation

```
src/tracker/
├── ml/
│   ├── __init__.py
│   ├── tcn.py              ← TCN architecture (PyTorch)
│   ├── embeddings.py       ← Training loop, embedding extraction
│   ├── similarity.py       ← Nearest-neighbor search, clustering
│   └── manifold.py         ← Dimensionality estimation, UMAP visualization
```

Dependencies: `torch`, `numpy`, `scikit-learn`, `umap-learn`

CLI additions:
```
clash-stats --train-embeddings          # Train/retrain embedding model
clash-stats --embed-all                 # Compute embeddings for all games with replay data
clash-stats --similar BATTLE_ID         # Find games similar to a specific battle
clash-stats --clusters                  # Show game clusters with win rates
clash-stats --manifold-viz              # Generate UMAP visualization
```

## Consequences

### Positive
- Enables every downstream model (ADR-004, 005, 006) to operate on learned representations instead of hand-crafted features
- Game similarity search is immediately useful for pattern recognition — "this loss looks like that loss"
- Manifold analysis reveals the intrinsic complexity of the game at your trophy range
- TCN architecture is small enough to train on CPU, retrain frequently as data accumulates
- Shared card embedding layer amortizes representation learning across all models

### Negative
- Requires PyTorch dependency (significant addition to the Docker image size)
- Model quality is bounded by data volume until the top-ladder pipeline (ADR-007) is active
- Embeddings must be recomputed when the model is retrained — stale embeddings from old model versions can mislead
- Contrastive training needs careful pair construction to avoid class imbalance (more wins than losses or vice versa)

### Scale Requirements

| Games with replay data | Training viability |
|----------------------|-------------------|
| < 100 | Insufficient — use Monte Carlo only (ADR-002) |
| 100-500 | Supervised win/loss classification only, expect 65-70% accuracy |
| 500-2,000 | Add contrastive learning, expect 75-80%, meaningful clusters emerge |
| 2,000-10,000 | Multi-task training, 80-85%, manifold geometry becomes informative |
| 10,000+ | Fine-grained sub-cluster analysis, per-card interaction embeddings, transfer to other models |
