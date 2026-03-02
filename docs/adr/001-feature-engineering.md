# ADR-001: Feature Engineering from Replay Events

**Status:** Implemented
**Date:** 2026-02-22
**Depends on:** Replay scraper (`replays.py`, migration 003)

## Context

The replay event stream captures card placements as `(side, card_name, game_tick, arena_x, arena_y, play_number, ability_used)` tuples, plus per-side elixir summaries. Raw events are not directly consumable by either statistical models or neural networks. We need a canonical feature extraction layer that transforms raw replay data into model-ready tensors and tabular features.

This ADR defines the feature engineering pipeline that all downstream models (ADR-002 through ADR-006) consume. Getting this right determines the ceiling for everything built on top.

## Decision

### 1. Per-Event Feature Vector

Each replay event becomes a feature vector with the following components:

| Feature | Type | Encoding | Dimension |
|---------|------|----------|-----------|
| `card_id` | Categorical | Learned embedding (all ~110 cards) | 16 |
| `side` | Binary | 0 = team, 1 = opponent | 1 |
| `game_tick` | Continuous | Normalized to [0, 1] over max game length (5280 ticks = 3:00 + 3:00 OT) | 1 |
| `game_phase` | Categorical | Regular / Double elixir / OT / OT double | 4 (one-hot) |
| `arena_x` | Continuous | Normalized to [0, 1] over arena width (18000 units) | 1 |
| `arena_y` | Continuous | Normalized to [0, 1] over arena height (32000 units) | 1 |
| `lane` | Categorical | Left / Right / Center (derived from arena_x) | 3 (one-hot) |
| `play_number` | Ordinal | Raw integer, capped at 10 | 1 |
| `ability_used` | Binary | 0/1 | 1 |
| `elixir_cost` | Continuous | Card's elixir cost (from card metadata table) | 1 |
| `card_type` | Categorical | Troop / Spell / Building | 3 (one-hot) |
| `is_evo` | Binary | Whether this card has evolution active | 1 |

**Total per-event dimension:** 34 (with 16-dim card embedding)

### 2. Derived Sequence Features

Computed over sliding windows of the event sequence:

| Feature | Window | Description |
|---------|--------|-------------|
| `elixir_spent_team` | Last 5 events | Running total elixir spent by team |
| `elixir_spent_opp` | Last 5 events | Running total elixir spent by opponent |
| `elixir_differential` | Last 5 events | Team - Opponent elixir spent |
| `play_rate_team` | Last 10 ticks | Events per tick (cycle speed proxy) |
| `play_rate_opp` | Last 10 ticks | Events per tick |
| `lane_pressure` | Current | Which lane has more active units (left/right/balanced) |
| `cards_remaining_team` | Current | Cards not yet seen in team's cycle |
| `cards_remaining_opp` | Current | Known cards not yet played in opponent's current cycle |
| `time_since_last_play` | Current | Ticks since this side's last event (elixir leak proxy) |

### 3. Per-Game Context Features

Static features attached to every event in a game:

| Feature | Description |
|---------|-------------|
| `player_deck_hash` | Hash of player's 8 cards (links to deck_cards table) |
| `opponent_deck_hash` | Hash of opponent's 8 cards |
| `avg_elixir_player` | Player deck average elixir cost |
| `avg_elixir_opponent` | Opponent deck average elixir cost |
| `elixir_differential_deck` | Deck cost difference (structural advantage/disadvantage) |
| `trophy_band` | Player trophy range bucket (10K-10.5K, 10.5K-11K, etc.) |
| `matchup_archetype` | Opponent archetype label from `archetypes.py` |

### 4. Card Metadata Table

A static lookup table mapping card names to properties. Stored as `card_metadata.json` in the data directory, version-controlled.

```json
{
  "pekka": {"elixir": 7, "type": "troop", "target": "ground", "speed": "slow", "range": "melee", "hp": 4480, "dps": 544},
  "miner": {"elixir": 3, "type": "troop", "target": "ground", "speed": "fast", "range": "melee", "hp": 1000, "dps": 160},
  ...
}
```

This is manually maintained. Card balance changes require updating this file. The alternative (scraping from RoyaleAPI or the game files) adds fragile dependencies for data that changes infrequently.

### 5. Elixir Curve Reconstruction

The replay events don't directly contain elixir state — they contain card plays with known costs and ticks. Elixir state is reconstructable:

```
elixir(t) = min(10, starting_elixir + generation_rate * t - sum(costs of cards played before t))
```

Where:
- `starting_elixir` = 5 (7 in some game modes)
- `generation_rate` = 1e per 2.8 seconds (1e per 1.4s in double elixir)
- Tick rate: ~30 ticks/second (derived from observed data, verify empirically)

**Elixir leak** is computable as the integral of `max(0, elixir(t) - 10)` over the game. The ReplaySummary table stores the ground-truth leaked value from RoyaleAPI, which serves as validation for the reconstructed curve.

### 6. Output Formats

The feature pipeline produces two output formats:

**Tabular (for Monte Carlo / statistical analysis):**
- One row per game with aggregated features
- Stored in a `game_features` SQLite table
- Columns: all per-game context features + aggregated event statistics

**Sequential (for neural models):**
- One tensor per game: `(num_events, feature_dim)` padded to max sequence length
- Stored as `.npz` files in a `features/` directory
- Includes attention mask for padded positions

### 7. Implementation

```
src/tracker/
├── features.py          ← Feature extraction from replay events
├── card_metadata.json   ← Static card properties
└── features/            ← Generated tensor cache (.npz)
```

`features.py` exposes:
- `extract_game_features(session, battle_id) -> dict` — Tabular features for one game
- `extract_sequence_features(session, battle_id) -> np.ndarray` — Tensor for one game
- `build_feature_cache(session, output_dir)` — Batch extraction for all games with replay data
- `get_card_embedding_matrix() -> np.ndarray` — Initial embedding weights (can be random or pretrained)

## Consequences

### Positive
- All downstream models share the same feature definitions — no feature drift between models
- Tabular output enables Monte Carlo analysis immediately without any ML infrastructure
- Sequential output is model-agnostic — works with LSTM, Transformer, 1D-CNN
- Elixir reconstruction from events gives continuous state, not just summary statistics
- Card embedding layer is shared across all neural models (pretrain once, reuse)

### Negative
- Card metadata table requires manual maintenance on balance patches
- Tick-to-second conversion factor needs empirical verification from replay data
- Feature cache must be rebuilt when feature definitions change
- Sequence padding wastes memory for short games (mitigated by bucketed batching)

### Risks
- If RoyaleAPI changes their replay HTML structure, the raw event data may lose fields, which would break derived features. Mitigation: the `raw_json` preservation pattern from the battle table should extend to replay HTML storage.
- Elixir reconstruction assumes standard game rules. Special game modes (draft, triple elixir, etc.) would produce incorrect curves. Mitigation: filter to `battle_type == "PvP"` for training data, which the scraper already does.
