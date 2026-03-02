# Machine Learning Pipeline — Technical Overview

**Version:** 1.0
**Date:** 2026-03-01
**Status:** Phase 0 (UMAP) and Phase 1 (TCN) implemented

## Abstract

This document describes the machine learning pipeline that transforms raw Clash Royale replay event telemetry into learned game state representations. The system operates in two implemented phases: Phase 0 applies classical dimensionality reduction (UMAP) to hand-crafted tabular feature vectors, while Phase 1 trains a Temporal Convolutional Network (TCN) on raw event sequences to produce learned 128-dimensional game embeddings. Both phases feed into a shared downstream analysis layer: HDBSCAN density-based clustering, Euclidean similarity search with Gaussian kernel scoring, three-leg manifold analysis via k-means macro-segmentation, and a two-layer tilt detection system combining heuristic signals with embedding-space proximity to known tilt cluster centroids.

The pipeline processes two data sources: personal games (~2,800 lifetime, ~1.35 games/day) and a top-ladder corpus (~10,000+ games from the global top-200 players, scraped via a combined battle-log + replay pipeline). Both share the same feature extraction, embedding, and analysis infrastructure with provenance tracking (`corpus` column) that enables corpus-wide pre-training with personal fine-tuning.

## System Architecture

```
                        ┌──────────────────────────────────────────────┐
                        │              Data Collection                 │
                        │                                              │
                        │  CR API (/battlelog)  ──►  battles table     │
                        │  RoyaleAPI (HTML)     ──►  replay_events     │
                        │                           replay_summaries   │
                        │                           deck_cards         │
                        └──────────────┬───────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                      │
            ┌───────▼───────┐                    ┌────────▼────────┐
            │  Phase 0       │                    │  Phase 1         │
            │  Tabular       │                    │  Sequential      │
            │  Features      │                    │  Features        │
            │                │                    │                  │
            │  50-dim        │                    │  17-dim/event    │
            │  per-game      │                    │  + card_id       │
            │  aggregates    │                    │  variable-len    │
            │                │                    │  sequences       │
            │  features.py   │                    │  sequence_       │
            │                │                    │  dataset.py      │
            └───────┬───────┘                    └────────┬────────┘
                    │                                      │
            ┌───────▼───────┐                    ┌────────▼────────┐
            │  UMAP          │                    │  TCN Encoder     │
            │  Pipeline      │                    │                  │
            │                │                    │  6-layer dilated │
            │  50→15→3 dim   │                    │  causal conv     │
            │  supervised    │                    │  128-dim output  │
            │  StandardScaler│                    │                  │
            │                │                    │  tcn.py          │
            │  umap_         │                    │  training.py     │
            │  embeddings.py │                    │                  │
            └───────┬───────┘                    └────────┬────────┘
                    │                                      │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │       Shared Analysis Layer          │
                    │                                      │
                    │  ┌────────────┐  ┌───────────────┐  │
                    │  │ HDBSCAN    │  │ Similarity    │  │
                    │  │ Clustering │  │ Search        │  │
                    │  │            │  │               │  │
                    │  │ clustering │  │ StandardScaler│  │
                    │  │ .py        │  │ + Euclidean   │  │
                    │  └────────────┘  │ + Gaussian    │  │
                    │                  │ kernel        │  │
                    │  ┌────────────┐  │               │  │
                    │  │ Manifold   │  │ similarity.py │  │
                    │  │ Analysis   │  └───────────────┘  │
                    │  │            │                      │
                    │  │ K-means    │  ┌───────────────┐  │
                    │  │ macro-legs │  │ Tilt          │  │
                    │  │ + temporal │  │ Detection     │  │
                    │  │ profiling  │  │               │  │
                    │  │            │  │ Heuristic +   │  │
                    │  │ cluster_   │  │ embedding     │  │
                    │  │ profiler.py│  │ proximity     │  │
                    │  └────────────┘  │               │  │
                    │                  │ tilt_          │  │
                    │                  │ detector.py   │  │
                    │                  └───────────────┘  │
                    └─────────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │         Storage Layer                 │
                    │                                      │
                    │  game_features    (50-dim BLOB)      │
                    │  game_embeddings  (128/15d + 3d)     │
                    │  battles, replay_events, deck_cards  │
                    │                                      │
                    │  storage.py — numpy ↔ SQLite BLOB    │
                    └──────────────────────────────────────┘
```

## Module Index

| Module | Lines | Purpose | Input | Output |
|--------|-------|---------|-------|--------|
| [`card_metadata.py`](../src/tracker/ml/card_metadata.py) | 137 | Dynamic card vocabulary from DB | `deck_cards` table | `CardVocabulary` (name↔index, elixir, type) |
| [`features.py`](../src/tracker/ml/features.py) | 283 | 50-dim tabular feature extraction | Replay events + battle metadata | `game_features` table (float32 BLOB) |
| [`umap_embeddings.py`](../src/tracker/ml/umap_embeddings.py) | 224 | Two-stage UMAP dimensionality reduction | 50-dim feature matrix | 15-dim analytical + 3-dim visualization embeddings |
| [`sequence_dataset.py`](../src/tracker/ml/sequence_dataset.py) | 245 | PyTorch Dataset for TCN training | Replay event sequences | Padded (card_id, features, label) tensors |
| [`tcn.py`](../src/tracker/ml/tcn.py) | 264 | TCN model architecture | (card_ids, features, lengths) | 128-dim embeddings + win logits |
| [`training.py`](../src/tracker/ml/training.py) | 550 | Training loop + inference pipeline | SequenceDataset + model config | Trained model + stored embeddings |
| [`clustering.py`](../src/tracker/ml/clustering.py) | 110 | HDBSCAN clustering + profiling | 15-dim or 128-dim embeddings | Cluster labels + per-cluster stats |
| [`similarity.py`](../src/tracker/ml/similarity.py) | 185 | k-NN similarity search | 50-dim feature vectors | Ranked similar games with metrics |
| [`cluster_profiler.py`](../src/tracker/ml/cluster_profiler.py) | 478 | Manifold leg analysis | TCN embeddings + replay events | Three-leg temporal profiles |
| [`tilt_detector.py`](../src/tracker/ml/tilt_detector.py) | 267 | Tilt pattern detection | Recent battles + embeddings | TiltStatus (level, metrics, message) |
| [`storage.py`](../src/tracker/ml/storage.py) | 53 | ORM models + serialization | numpy arrays | SQLite BLOBs |

## Detailed Documentation

- [Feature Engineering](./ml-feature-engineering.md) — 50-dim tabular vectors and 17-dim per-event sequential features
- [TCN Architecture](./ml-tcn-architecture.md) — Temporal Convolutional Network design, training, and inference
- [Dimensionality Reduction & Clustering](./ml-umap-clustering.md) — UMAP pipeline, HDBSCAN, manifold analysis
- [Similarity Search & Tilt Detection](./ml-similarity-tilt.md) — Distance metrics, kernel methods, tilt heuristics
- [Empirical Findings](./ml-empirical-findings.md) — Three-leg manifold structure, high-Z outliers, player comparisons

## Data Flow Summary

### Phase 0 (UMAP) Pipeline

```
replay_events + replay_summaries + deck_cards + battles
  │
  ▼  features.py::extract_game_features()
  │
  50-dim float32 vector per game
  │  Stored: game_features table (BLOB)
  │
  ▼  umap_embeddings.py::EmbeddingPipeline.fit_transform()
  │
  StandardScaler → UMAP(50→15, supervised) → UMAP(15→3)
  │  Stored: game_embeddings table (15d + 3d BLOBs)
  │
  ▼  clustering.py::label_clusters()
  │
  HDBSCAN(min_cluster_size=10, min_samples=5, eom)
  │  Stored: game_embeddings.cluster_id
  │
  ▼  Dashboard: 3D Plotly.js scatter plot
```

### Phase 1 (TCN) Pipeline

```
replay_events + deck_cards
  │
  ▼  sequence_dataset.py::SequenceDataset
  │
  Per-event: card_id (int64) + 17-dim features (float32)
  Variable-length sequences, padded per batch
  │
  ▼  tcn.py::GameEmbeddingModel.forward()
  │
  Card embedding(16d) ⊕ features(17d) = 33d per event
  → TCN(6 blocks, dilations [1,2,4,8,16,32])
  → Masked pooling (mean ⊕ max ⊕ last = 768d)
  → Projection (768→256→128)
  │
  128-dim game embedding + win probability logit
  │
  ▼  training.py::train_tcn()
  │
  BCEWithLogitsLoss, AdamW, CosineAnnealingLR
  Early stopping (patience=10)
  │
  ▼  umap_embeddings.py::reduce_to_3d()
  │
  UMAP(128→3) for visualization
  │
  ▼  clustering.py::label_clusters()
  │
  HDBSCAN on 128-dim space
  │
  ▼  Dashboard: 3D Plotly.js scatter plot + manifold analysis
```

### Incremental Embedding Pipeline

```
New games with replay data (post-training)
  │
  ▼  training.py::embed_new()
  │
  Load saved tcn_v1.pt + umap_3d_standalone.pkl
  Forward pass (no gradient) → 128-dim embeddings
  UMAP transform() (no refit) → 3-dim coordinates
  │
  Stored: game_embeddings (cluster_id = NULL for incremental)
```

## Hardware

- **Training host:** Intel Xeon (12-core) + Intel Arc A770 (16GB VRAM)
- **Device detection priority:** XPU (Intel oneAPI) → CUDA → CPU
- **Training time:** ~2 minutes for 10K games on CPU; seconds on XPU
- **Model size:** ~2M parameters (TCN), ~50MB on disk
- **Inference:** ~0.5ms per game on CPU
