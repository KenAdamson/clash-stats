# Dimensionality Reduction, Clustering & Manifold Analysis

**Version:** 1.0
**Date:** 2026-03-01
**Implements:** ADR-003 Phase 0 (UMAP) + downstream clustering and manifold analysis

## 1. UMAP: Theoretical Foundation

UMAP (Uniform Manifold Approximation and Projection; McInnes, Healy, & Melville, 2018) is a manifold learning algorithm grounded in algebraic topology. Unlike t-SNE, which optimizes KL divergence between high-dimensional and low-dimensional probability distributions, UMAP constructs a topological representation (a fuzzy simplicial set) in both spaces and minimizes their cross-entropy.

### 1.1 Algorithm Summary

1. **Construct a weighted k-nearest-neighbor graph** in high-dimensional space. Edge weights decay exponentially with distance from each point's nearest neighbor, normalized so that each point's local connectivity integrates to $\log_2(k)$.

2. **Symmetrize the graph** by taking the probabilistic union: $w_{ij} = w_{i \to j} + w_{j \to i} - w_{i \to j} \cdot w_{j \to i}$.

3. **Initialize a low-dimensional layout** (spectral initialization by default).

4. **Optimize the low-dimensional layout** to minimize cross-entropy between the high-dimensional fuzzy simplicial set and the low-dimensional one, using stochastic gradient descent with negative sampling.

### 1.2 Key Properties for This Application

- **Metrically meaningful distances:** Unlike t-SNE, UMAP preserves both local neighborhood structure *and* global distances (to the extent permitted by the dimensionality reduction). Points that are far apart in embedding space genuinely represent dissimilar games.

- **Supervised mode:** The `target_metric='categorical'` parameter injects label information (win/loss) into the graph construction. Points with the same label receive additional weight in the neighbor graph, encouraging same-label points to cluster together *if the feature space supports it*. The manifold structure that emerges isn't imposed — it's the structure in the data that correlates with labels.

- **Incremental `transform()`:** New data points can be projected into a fitted embedding space via `reducer.transform(new_data)`, which finds each point's neighbors in the training data and optimizes its position relative to them. This is an approximation — the projection quality degrades as new data drifts from the training distribution.

## 2. Two-Stage UMAP Pipeline

**Implementation:** `src/tracker/ml/umap_embeddings.py::EmbeddingPipeline`

### 2.1 Phase 0 Pipeline: 50 → 15 → 3

```
Raw features (50-dim)
  │
  ▼  StandardScaler.fit_transform()
  │
  Standardized features (50-dim, μ=0, σ=1 per dimension)
  │
  ▼  UMAP(n_components=15, n_neighbors=30, min_dist=0.0,
  │       metric='euclidean', target_metric='categorical')
  │       with y=win_loss_labels (supervised)
  │
  Analytical embedding (15-dim) — for downstream computation
  │
  ▼  UMAP(n_components=3, n_neighbors=30, min_dist=0.3,
  │       spread=1.5, metric='euclidean')
  │
  Visualization embedding (3-dim) — for 3D scatter plots
```

### 2.2 Phase 1 Pipeline: 128 → 3

When TCN embeddings are available, the pipeline projects the learned 128-dim space to 3D:

```
TCN embeddings (128-dim)
  │
  ▼  UMAP(n_components=3, n_neighbors=30, min_dist=0.3,
  │       spread=1.5, metric='euclidean')
  │
  Visualization embedding (3-dim)
```

This uses `EmbeddingPipeline.reduce_to_3d()`, which fits a standalone UMAP reducer (saved as `umap_3d_standalone.pkl`) for incremental transform of new embeddings.

### 2.3 Hyperparameter Choices

| Parameter | Stage 1 (50→15) | Stage 2 (15→3 or 128→3) | Rationale |
|-----------|-----------------|-------------------------|-----------|
| `n_components` | 15 | 3 | 15-dim preserves enough structure for clustering; 3-dim for human visualization |
| `n_neighbors` | 30 | 30 | Balances local detail vs global structure. With ~10K games, 30 neighbors captures enough context without over-smoothing |
| `min_dist` | 0.0 | 0.3 | Stage 1: allow tight clustering (0.0 permits points to overlap). Stage 2: spread points for visualization (0.3 prevents visual clumping) |
| `spread` | 1.0 (default) | 1.5 | Wider spread in visualization stage for better visual separation |
| `metric` | euclidean | euclidean | Standard for standardized continuous features |
| `random_state` | 42 | 42 | Reproducibility |

### 2.4 StandardScaler Normalization

UMAP is sensitive to feature scale — a feature ranging [0, 100] will dominate distance calculations over a feature ranging [0, 1]. The `StandardScaler` transforms each dimension to zero mean and unit variance:

$$x'_j = \frac{x_j - \mu_j}{\sigma_j}$$

This ensures that all 50 feature dimensions contribute proportionally to the distance metric. The fitted scaler is saved with the pipeline for consistent transform of new data.

### 2.5 Model Persistence

The `EmbeddingPipeline` saves its state as a pickled dictionary:

```python
{
    "scaler": StandardScaler (fitted),
    "reducer_15d": UMAP (fitted),
    "reducer_3d": UMAP (fitted),
}
```

Saved to: `data/ml_models/umap_pipeline.pkl`

For TCN-based 3D projection, a standalone reducer is saved separately: `data/ml_models/umap_3d_standalone.pkl`

---

## 3. HDBSCAN Clustering

**Implementation:** `src/tracker/ml/clustering.py`

### 3.1 Why HDBSCAN

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise; Campello, Moulavi, & Sander, 2013) is chosen over k-means, DBSCAN, and agglomerative clustering for the following reasons:

| Property | HDBSCAN | K-Means | DBSCAN |
|----------|---------|---------|--------|
| Number of clusters | Automatic | Requires $k$ a priori | Automatic |
| Cluster shapes | Arbitrary | Spherical (Voronoi cells) | Arbitrary |
| Varying density | Handles naturally | Cannot | Struggles (global ε) |
| Noise detection | Built-in (label -1) | No — every point assigned | Built-in |
| Deterministic | Nearly (approximate) | Random initialization | Yes |

The key advantage: HDBSCAN discovers clusters of varying density and explicitly labels outliers. In game embedding space, some clusters are dense (many similar games, e.g., "standard Golem beatdown wins") while others are sparse (rare game types). HDBSCAN handles both naturally.

### 3.2 Parameters

```python
HDBSCAN_PARAMS = dict(
    min_cluster_size=10,          # minimum games to form a cluster
    min_samples=5,                # conservative density estimate
    cluster_selection_method="eom",  # Excess of Mass
)
```

- **`min_cluster_size=10`:** A cluster must contain at least 10 games. This prevents trivial clusters from individual game quirks. With ~10K games, this produces 50-80 clusters.

- **`min_samples=5`:** Controls the conservativeness of the density estimate. Lower values create more clusters; higher values require denser concentrations. 5 is a moderate choice.

- **`cluster_selection_method="eom"`:** Excess of Mass (EOM) selects clusters from the condensed tree by maximizing total cluster "mass" (the integral of persistence over the lifetime of each cluster in the hierarchy). This produces more stable clusters than the alternative "leaf" method, which tends to over-segment.

### 3.3 Algorithm Sketch

1. Compute mutual reachability distance: $d_{\text{mreach}}(a, b) = \max(\text{core}_k(a), \text{core}_k(b), d(a, b))$ where $\text{core}_k$ is the distance to the $k$-th nearest neighbor.
2. Build a minimum spanning tree on the mutual reachability graph.
3. Construct a hierarchical clustering from the MST.
4. Condense the hierarchy using `min_cluster_size`.
5. Extract flat clusters via the EOM algorithm.

### 3.4 Cluster Profiling

**Implementation:** `src/tracker/ml/clustering.py::profile_clusters()`

For each cluster, the profiler computes:
- **Size:** Number of games
- **Win rate:** Fraction of wins
- **Personal count:** Games from the personal corpus (vs. top-ladder corpus)
- **3D centroid:** Mean of UMAP 3D coordinates (for visualization positioning)

---

## 4. Three-Leg Manifold Analysis

**Implementation:** `src/tracker/ml/cluster_profiler.py`

### 4.1 The "Elder God" Structure

When TCN embeddings are projected to 3D via UMAP, the resulting point cloud consistently exhibits a three-armed structure — three elongated legs radiating from a central core. This structure is not imposed by the analysis; it emerges from the data.

The three legs correspond to qualitatively different game states:

| Leg | Name | Win Rate | Aggression | Character |
|-----|------|----------|------------|-----------|
| 0 | Dominant | ~62% | ~67% | Controlling the game, plays in opponent's half |
| 1 | Contested | ~54% | ~57% | Even exchanges, competitive games |
| 2 | Overwhelmed | ~50% | ~34% | Defensive, pushed back into own half |

### 4.2 Macro-Leg Detection: K-Means

**Implementation:** `cluster_profiler.py::identify_legs()`

The three legs are identified by running k-means ($k=3$) on the 3D UMAP coordinates:

```python
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(coords_3d)  # (n_games, 3)
```

After clustering, legs are **reordered by win rate** (highest → leg 0, lowest → leg 2) for consistent naming across runs:

```python
sorted_legs = sorted(leg_wr.keys(), key=lambda k: -leg_wr[k])
# leg 0 = dominant, leg 1 = contested, leg 2 = overwhelmed
```

**Why k-means on 3D coordinates (not 128D embeddings):**
The three-leg structure is a visual/topological phenomenon that manifests in the UMAP projection. Running k-means on the 3D projection captures the visually apparent arms. Running it on 128D would produce different (and less interpretable) groupings because the 128D space has structure that the 3D projection collapses.

### 4.3 Temporal Profiling: `_profile_group()`

For each leg (or any group of games), the profiler computes a rich feature profile from replay event data:

**Temporal features:**
- Game phase distribution (fraction of events in regular/double/overtime/OT-double)
- Average phase fraction per game (accounts for variable game lengths)
- First and last play tick (proxy for game duration in replay terms)

**Spatial features:**
- Mean arena Y for team and opponent plays
- Aggression index: fraction of team plays in opponent's half
- Lane distribution (left/right/center)

**Card type features:**
- Troop/spell/building ratios (fraction of all events)

**Tempo features:**
- Average plays per game
- Median and mean inter-play tick gaps
- Alternation rate: fraction of consecutive events that switch sides (team→opponent or vice versa). High alternation indicates reactive play; low indicates committed sequences.

**Economy features:**
- Average player and opponent elixir leaked
- Crown differential

**Top cards:**
- 8 most frequently played team and opponent cards

### 4.4 Manifold Profile Comparison

**Implementation:** `cluster_profiler.py::profile_manifold()`

Generates comparative insights between the dominant and overwhelmed legs:
- Win rate spread
- Tempo difference (plays per game)
- Aggression gap
- Alternation rate (reactive vs committed)
- Phase distribution shifts (where in the game are plays concentrated)
- Elixir leak comparison
- Card type shifts (spell/building usage differences)

**Example output (from live analysis, 10,613 games):**

```
Win rate spread: dominant 62.5% vs overwhelmed 50.5% (12.0% gap)
Tempo: dominant 26 plays/game, overwhelmed 24 plays/game
Aggression: dominant 67.2% offensive, overwhelmed 34.1% offensive
Alternation: dominant 55.3%, overwhelmed 52.8% (more reactive)
Elixir leak: dominant 2.3e, overwhelmed 4.1e
```

The aggression index is the most discriminative feature: a 33-percentage-point gap between dominant (67%) and overwhelmed (34%) legs. This makes physical sense — when you control the game, your card placements are in the opponent's half of the arena; when you're being overwhelmed, they're in yours.

### 4.5 High-Z Outlier Phenomenon

In the UMAP 3D projection, a distinct cluster of games appears at high values of the third coordinate (Z > 25). Investigation reveals these are **elite stalemates** — games between 12,000+ trophy players who cannot break each other:

- 112 games, all corpus (zero personal)
- 79% single-crown outcomes (1-0 or 0-1)
- 58% of games extend past tick 5000 (overtime)
- Median player trophies: 12,500
- Average total plays per game: 40.7 (vs 25.4 for the main body)
- HDBSCAN assigns them to a single cluster (cluster 58)

These games occupy a topologically distinct region of the manifold because their event sequences have a qualitatively different structure: long, dense, evenly-matched exchanges with no decisive breakthrough.
