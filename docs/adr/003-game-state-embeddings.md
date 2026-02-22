# ADR-003: Game State Embedding Model

**Status:** Proposed
**Date:** 2026-02-22
**Depends on:** ADR-001 (Feature Engineering), ADR-007 (Training Data Pipeline)

## Context

A game of Clash Royale is a trajectory through a high-dimensional state space. Two games can look superficially different (different cards, different positions, different timing) but be structurally identical (both are "opponent overcommits left lane, player punishes with opposite-lane push, snowballs into three-crown"). Current analysis treats each game as an isolated data point. Embeddings allow us to discover the latent structure — the manifold of game trajectories that separates wins from losses, aggression from defense, optimal play from misplay.

## Decision

### Phased Approach

The embedding system is built in two phases. Phase 0 (UMAP) runs immediately on current data with no training infrastructure. Phase 1 (TCN) upgrades to learned representations when the corpus reaches sufficient scale. Both phases produce embeddings that downstream models can consume — the interface is identical, only the embedding method changes.

### 1. Phase 0: UMAP Dimensionality Reduction (Available Now)

UMAP (Uniform Manifold Approximation and Projection) is not a visualization gimmick — it's a manifold learning algorithm grounded in Riemannian geometry. It constructs a weighted k-nearest-neighbor graph in high-dimensional space, models it as a fuzzy simplicial complex, and optimizes a low-dimensional projection that preserves the topological structure. Unlike t-SNE, UMAP embeddings are metrically meaningful: distances in the reduced space correspond to meaningful differences in the original data.

**Why UMAP is the right Phase 0:**

| Property | UMAP | Neural (TCN) |
|----------|------|-------------|
| Training data minimum | 50 games | 500+ games |
| Training time | Seconds | Minutes to hours |
| GPU required | No | No (but helps) |
| Dependencies | `umap-learn`, `scikit-learn` | `torch` (500MB+ Docker image delta) |
| Embedding quality at 200 games | Good — preserves local + global structure | Poor — insufficient training signal |
| Supervised variant | Yes (`target_metric` parameter) | Yes (classification head) |
| Incremental embedding of new games | `transform()` on fitted model | Forward pass through trained network |

**Method:**

1. Build the tabular feature vector per game from ADR-001: aggregate replay events into a fixed-width representation. Per game, this includes:
   - Per-card play counts and average positions (8 cards × 3 features = 24 dims)
   - Elixir economy features: total spent, leaked, troop/spell/building split ratios (10 dims)
   - Tempo features: events per minute, average inter-event gap, play rate by game phase (8 dims)
   - Outcome-adjacent features: crown differential, game duration, overtime flag (3 dims)
   - Matchup context: opponent avg elixir, archetype encoding (5 dims)
   - **Total: ~50-dimensional raw feature vector per game**

2. Normalize features (StandardScaler — UMAP is sensitive to feature scale)

3. UMAP reduction in two stages:
   ```
   Raw features (50-dim)
     │
     ▼
   UMAP(n_components=15, metric='euclidean', n_neighbors=15, min_dist=0.1)
     │
     ▼
   Analytical embedding (15-dim) — for downstream computation
     │
     ▼
   UMAP(n_components=2, metric='euclidean')
     │
     ▼
   Visualization embedding (2-dim) — for plotting
   ```

   The 15-dim analytical embedding is the workhorse. It feeds into similarity search, clustering, and the win probability model. The 2-dim projection is for human consumption only.

4. **Supervised UMAP** for win/loss separation:
   ```python
   import umap
   reducer = umap.UMAP(n_components=15, target_metric='categorical')
   embedding = reducer.fit_transform(features, y=win_loss_labels)
   ```
   The `target_metric='categorical'` parameter informs the projection with win/loss labels, so the reduced space naturally separates wins from losses without a classification head. The manifold structure that emerges isn't imposed by the labels — it's the structure in the data that *correlates with* the labels. Subtle but critical distinction.

5. **HDBSCAN clustering** on the 15-dim embedding:
   ```python
   import hdbscan
   clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
   labels = clusterer.fit_predict(embedding_15d)
   ```
   HDBSCAN over k-means because it finds clusters of varying density and identifies outliers (noise points = games that don't fit any pattern). The clusters that emerge are the natural "game types" at your trophy range.

6. **Incremental embedding:** When new games come in, use `reducer.transform(new_features)` to project them into the existing embedding space without refitting the entire model. Periodic refitting (weekly or after 50 new games) updates the manifold structure.

**What UMAP reveals immediately with 200 games:**

- **Win/loss topology:** Do wins and losses occupy distinct regions, or are they interleaved? Distinct regions = matchup-determined outcomes. Interleaved = execution-determined outcomes. This answers the fundamental question: "Am I losing because of my deck or because of my play?"
- **Failure mode clusters:** Losses that cluster together share structural features. "These 8 losses all cluster in the same region — what do they have in common?" Maybe they're all Inferno Dragon matchups. Maybe they're all games where you leaked 5+ elixir in the first minute. The clusters surface the pattern.
- **Outlier games:** Games far from any cluster are anomalies. These are the ones worth manual review — either novel opponent strategies or your own unusual plays.
- **Archetype neighborhoods:** Games against the same archetype should cluster. If they don't, the archetype classification from `archetypes.py` is too coarse — the UMAP embedding reveals sub-archetypes.

### 2. Phase 1: Temporal Convolutional Network (TCN) — Neural Upgrade Path

When the corpus reaches 500+ games with replay data (or immediately if the top-ladder pipeline from ADR-007 is active), the TCN replaces UMAP as the primary embedding method. The TCN operates on the raw event *sequence* rather than the aggregated tabular features, capturing temporal dynamics that UMAP on aggregated features cannot.

The game event sequence is an ordered time series of variable length (20-100+ events per game). The TCN:
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

**UMAP → TCN transition:** When the TCN is trained, compute both UMAP and TCN embeddings in parallel for a validation period. Compare downstream task performance (clustering coherence, similarity search relevance, win prediction accuracy). Switch to TCN as primary when it demonstrably outperforms UMAP. Keep UMAP as a diagnostic — applying UMAP to TCN embeddings reveals whether the neural model has learned a better manifold than raw features.

### 3. TCN Model Structure

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

### 4. TCN Training Objectives

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

### 5. Embedding Applications

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

### 6. Embedding Storage

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

### 7. Manifold Geometry

The 128-dim embedding space will not be uniformly occupied — game trajectories live on a lower-dimensional manifold within it. Understanding the manifold geometry is where Ken's experience with high-dimensional manifold generation becomes directly applicable:

- **Intrinsic dimensionality estimation:** Use methods like MLE (Levina-Bickel) or TwoNN to estimate the true dimensionality of the game manifold. If 128-dim embeddings live on a 12-dim manifold, that tells us there are ~12 independent "factors of variation" in how games unfold.
- **Manifold interpolation:** Given two game embeddings (e.g., a clean win and a close loss against the same archetype), interpolate along the geodesic on the manifold. The intermediate points represent hypothetical games that "bridge" the two outcomes. If a decoder exists (ADR-006), these can be decoded into synthetic event sequences.
- **Curvature analysis:** Regions of high curvature on the manifold correspond to decision boundaries — small changes in play lead to large changes in outcome. These are the critical moments in a game. Identifying high-curvature regions tells you *where* the game is most sensitive to player decisions.
- **Density estimation:** Fit a normalizing flow or GMM to the embedding distribution. Low-density regions represent rare game types. If a particular matchup pushes games into low-density regions, that matchup is genuinely unusual and can't be modeled by interpolation from common patterns.

### 8. Implementation

```
src/tracker/
├── ml/
│   ├── __init__.py
│   ├── umap_embeddings.py  ← Phase 0: UMAP feature aggregation, reduction, clustering
│   ├── tcn.py              ← Phase 1: TCN architecture (PyTorch)
│   ├── embeddings.py       ← Training loop, embedding extraction (both phases)
│   ├── similarity.py       ← Nearest-neighbor search, clustering
│   └── manifold.py         ← Dimensionality estimation, visualization
```

Dependencies:
- Phase 0: `umap-learn`, `hdbscan`, `scikit-learn`, `numpy` (lightweight, no PyTorch)
- Phase 1: adds `torch`

CLI additions:
```
# Phase 0 (available now)
clash-stats --umap-embed                # Compute UMAP embeddings for all games with replay data
clash-stats --umap-supervised           # Supervised UMAP with win/loss labels
clash-stats --similar BATTLE_ID         # Find games similar to a specific battle
clash-stats --clusters                  # Show game clusters with win rates
clash-stats --manifold-viz              # Generate 2D UMAP visualization
clash-stats --outliers                  # Surface anomalous games

# Phase 1 (when corpus scale supports it)
clash-stats --train-embeddings          # Train/retrain TCN embedding model
clash-stats --embed-all                 # Compute TCN embeddings for all games
clash-stats --compare-embeddings        # UMAP vs TCN quality comparison
```

## Consequences

### Positive
- **Phase 0 is immediately actionable** — UMAP on 200 games produces meaningful embeddings today, no training infrastructure required
- Supervised UMAP separates wins from losses using manifold structure that *correlates with* labels, not structure *imposed by* labels
- HDBSCAN discovers natural game clusters without specifying cluster count — reveals structure you wouldn't think to look for
- Phase 0 → Phase 1 transition is seamless — same embedding storage schema, same downstream interfaces
- UMAP on TCN embeddings serves as a diagnostic for the neural model quality
- Enables every downstream model (ADR-004, 005, 006) to operate on learned representations instead of hand-crafted features
- Game similarity search is immediately useful for pattern recognition — "this loss looks like that loss"
- Manifold analysis reveals the intrinsic complexity of the game at your trophy range
- Shared card embedding layer (Phase 1) amortizes representation learning across all models

### Negative
- UMAP embeddings from aggregated features lose temporal ordering — "Pekka first then Miner" and "Miner first then Pekka" produce identical tabular features. This is the fundamental limitation that the TCN upgrade addresses.
- UMAP's `transform()` for new points is approximate — the embedding of a new game depends slightly on the training set composition. Periodic refitting is necessary.
- Phase 1 requires PyTorch dependency (significant addition to the Docker image size)
- TCN model quality is bounded by data volume until the top-ladder pipeline (ADR-007) is active
- Embeddings must be recomputed when models are retrained — stale embeddings from old model versions can mislead
- Contrastive training (Phase 1) needs careful pair construction to avoid class imbalance

### Scale Requirements

| Games with replay data | Phase | Capability |
|----------------------|-------|------------|
| 50 | Phase 0 | UMAP with unsupervised clustering — discover game types |
| 100 | Phase 0 | Supervised UMAP — win/loss separation, failure mode clusters |
| 200 | Phase 0 | **Current scale.** Full UMAP pipeline: similarity search, archetype fingerprinting, anomaly detection, manifold visualization |
| 500 | Phase 0→1 | TCN training viable — run both in parallel, compare quality |
| 500-2,000 | Phase 1 | TCN with contrastive learning, expect 75-80% classification, meaningful sequence-aware clusters emerge |
| 2,000-10,000 | Phase 1 | Multi-task TCN training, 80-85%, manifold geometry becomes informative |
| 10,000+ | Phase 1 | Fine-grained sub-cluster analysis, per-card interaction embeddings, transfer to other models |
