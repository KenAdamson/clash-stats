# Architecture Decision Records — ML & Simulation Layer

This directory contains ADRs for the machine learning and Monte Carlo simulation capabilities built on top of the replay event data pipeline.

## Context

The replay scraper (`replays.py`) captures every card placement event with tick-level timing and arena coordinates, plus per-side elixir economy summaries. This transforms the tracker from a results database into a full game telemetry system. These ADRs describe how to exploit that telemetry for predictive modeling and strategic analysis.

## ADR Index

| ADR | Title | Status | Summary |
|-----|-------|--------|---------|
| [001](001-feature-engineering.md) | Feature Engineering from Replay Events | Proposed | Canonical feature extraction pipeline that all downstream models consume |
| [002](002-monte-carlo-simulation.md) | Monte Carlo Simulation Framework | Proposed | Elixir economy modeling, opening hand analysis, matchup probability distributions |
| [003](003-game-state-embeddings.md) | Game State Embedding Model | Proposed | Learned vector representations of game states via contrastive/supervised training |
| [004](004-win-probability-estimator.md) | Real-Time Win Probability Estimator | Proposed | P(win) at any tick given game state history — the "WPA" of Clash Royale |
| [005](005-opponent-prediction.md) | Opponent Play Prediction Model | Proposed | Sequence model predicting next card/position/timing from opponent |
| [006](006-counterfactual-simulator.md) | Counterfactual Deck Simulator | Proposed | Generative model for synthetic game sequences under deck modifications |
| [007](007-training-data-pipeline.md) | Training Data Pipeline & Scale Strategy | Proposed | Scaling beyond personal replays to top-ladder corpus via extended scraping |

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
 └──► 006 Counterfactual Simulator (depends on 003 + 005)
```

## Guiding Principles

1. **Statistical models first, deep learning second.** Monte Carlo runs on day one with 200 games. Neural models need scale. Build the pipeline so both consume the same features.
2. **Everything is queryable.** Model outputs go back into SQLite. Predictions are data, not just console output.
3. **Train on the meta, fine-tune on you.** General models from top-ladder corpus, specialized via transfer learning on KrylarPrime's games.
4. **Offline-first.** All training and inference runs locally on idle compute. No cloud dependencies, no API costs for inference.
