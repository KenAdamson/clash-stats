# Architecture Decision Records — ML & Simulation Layer

This directory contains ADRs for the machine learning and Monte Carlo simulation capabilities built on top of the replay event data pipeline.

## Context

The replay scraper (`replays.py`) captures every card placement event with tick-level timing and arena coordinates, plus per-side elixir economy summaries. This transforms the tracker from a results database into a full game telemetry system. These ADRs describe how to exploit that telemetry for predictive modeling and strategic analysis.

## ADR Index

| ADR | Title | Status | Summary |
|-----|-------|--------|---------|
| [001](001-feature-engineering.md) | Feature Engineering from Replay Events | Implemented | Canonical feature extraction pipeline that all downstream models consume |
| [002](002-monte-carlo-simulation.md) | Monte Carlo Simulation Framework | Proposed | Elixir economy modeling, opening hand analysis, matchup probability distributions |
| [003](003-game-state-embeddings.md) | Game State Embedding Model | Phase 0+1 Implemented | Learned vector representations of game states via UMAP and TCN |
| [004](004-win-probability-estimator.md) | Real-Time Win Probability Estimator | **Implemented (v2)** | P(win) at every tick, WPA per card, Platt-calibrated (78.4% acc, ECE=0.031) |
| [005](005-opponent-prediction.md) | Opponent Play Prediction Model | Proposed | Sequence model predicting next card/position/timing from opponent |
| [006](006-counterfactual-simulator.md) | Counterfactual Deck Simulator | Proposed | Generative model for synthetic game sequences under deck modifications |
| [007](007-training-data-pipeline.md) | Training Data Pipeline & Scale Strategy | Implemented | Scaling beyond personal replays to top-ladder corpus via extended scraping |
| [008](008-observability.md) | Observability | — | — |
| [009](009-visual-game-state-recognition.md) | Visual Game State Recognition | Proposed | LLM-bootstrapped CV pipeline: Claude labels → DINOv2 embeddings → YOLO distillation → full tactical analysis |

## Detailed Technical Documentation

In addition to the ADRs, detailed implementation documentation is available in `docs/`:

| Document | Description |
|----------|-------------|
| [ML Pipeline Overview](../ml-pipeline-overview.md) | System architecture, module index, data flow diagrams |
| [Feature Engineering](../ml-feature-engineering.md) | 50-dim tabular and 17-dim sequential feature specifications |
| [TCN Architecture](../ml-tcn-architecture.md) | Temporal Convolutional Network design, training loop, inference |
| [UMAP & Clustering](../ml-umap-clustering.md) | Dimensionality reduction, HDBSCAN, manifold analysis |
| [Similarity & Tilt](../ml-similarity-tilt.md) | Distance metrics, Gaussian kernel similarity, tilt detection |
| [Empirical Findings](../ml-empirical-findings.md) | Three-leg structure, high-Z outliers, player comparisons |

## Dependency Graph

```
007 Training Data Pipeline
 │
 ▼
001 Feature Engineering ◄── All models consume this
 │
 ├──► 002 Monte Carlo Simulation (statistical, no ML training)
 │
 ├──► 003 Game State Embeddings
 │     │
 │     ├──► 004 Win Probability Estimator
 │     │
 │     └──► 005 Opponent Prediction
 │
 ├──► 006 Counterfactual Simulator (depends on 003 + 005)
 │
 └──► 009 Visual Game State Recognition
       │   (Claude Vision → DINOv2 → YOLO → full arena state)
       │
       ├──► 003 Game State Embeddings (visual features replace/augment replay features)
       ├──► 004 Win Probability (per-tick ArenaState input)
       └──► 005 Opponent Prediction (spatial context for placement prediction)
```

## Guiding Principles

1. **Statistical models first, deep learning second.** Monte Carlo runs on day one with 200 games. Neural models need scale. Build the pipeline so both consume the same features.
2. **Everything is queryable.** Model outputs go back into SQLite. Predictions are data, not just console output.
3. **Train on the meta, fine-tune on you.** General models from top-ladder corpus, specialized via transfer learning on KrylarPrime's games.
4. **Offline-first.** All training and inference runs locally on idle compute. No cloud dependencies, no API costs for inference.
