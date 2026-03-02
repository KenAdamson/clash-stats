# Feature Engineering — Mathematical Specification

**Version:** 1.0
**Date:** 2026-03-01
**Implements:** ADR-001, ADR-003 Phase 0 & Phase 1

## Overview

The pipeline extracts two parallel feature representations from the same replay event data:

1. **Tabular features** (Phase 0): A fixed-width 50-dimensional vector per game, computed by aggregating replay events into summary statistics. Used by UMAP, similarity search, and statistical analysis.

2. **Sequential features** (Phase 1): A variable-length sequence of 17-dimensional vectors (one per replay event), paired with integer card IDs for learned embedding. Used by the TCN model.

Both representations share the same raw data source — the `replay_events`, `replay_summaries`, `deck_cards`, and `battles` tables — but encode it at different granularities. The tabular representation discards temporal ordering (a fundamental limitation), while the sequential representation preserves it.

## 1. Tabular Feature Vector (50-dim)

**Implementation:** `src/tracker/ml/features.py::extract_game_features()`
**Storage:** `game_features` table, `feature_vector` column (float32 BLOB, 200 bytes)
**Versioning:** `feature_version = "v1"` — bump when extraction logic changes

### 1.1 Card Play Counts (24 dimensions)

The first 24 dimensions encode card usage frequency, split by side and play type.

**Player card play counts** (dims 0-7):
For each of the player's 8 deck cards (sorted alphabetically by `card_name`), count the number of non-ability plays by the team side:

$$x_i = |\{e \in E_{\text{team}} : e.\text{card\_name} = c_i \land \neg e.\text{ability\_used}\}|, \quad i \in [0, 7]$$

where $E_{\text{team}}$ is the set of team-side replay events and $c_i$ is the $i$-th player card in sorted order. If fewer than 8 cards are present (shouldn't happen in standard PvP), remaining dimensions are zero-padded.

**Opponent card play counts** (dims 8-15): Identical logic applied to opponent-side events and opponent deck cards.

**Player ability uses** (dims 16-19): Count of evolution/hero ability activations, grouped by distinct card name (up to 4 ability-bearing cards):

$$x_{16+j} = |\{e \in E_{\text{team}} : e.\text{card\_name} = a_j \land e.\text{ability\_used}\}|, \quad j \in [0, 3]$$

where $a_j$ is the $j$-th distinct ability card name in sorted order.

**Opponent ability uses** (dims 20-23): Same logic for opponent side.

**Design note:** Card play counts are raw integers, not normalized. The StandardScaler applied before UMAP and similarity search handles normalization. Raw counts preserve more information than fractions — a game with 3 plays of a card out of 15 total is different from 6 plays out of 30, even though the fraction is identical.

### 1.2 Elixir Economy (10 dimensions)

These features encode resource management efficiency, extracted from the `replay_summaries` table.

| Dim | Feature | Formula | Range |
|-----|---------|---------|-------|
| 24 | Total elixir spent (player) | `team_summary.total_elixir` | [0, ~100] |
| 25 | Total elixir spent (opponent) | `opp_summary.total_elixir` | [0, ~100] |
| 26 | Elixir differential | `total_player - total_opponent` | [-100, 100] |
| 27 | Player elixir leaked | `battle.player_elixir_leaked` | [0, ~30] |
| 28 | Opponent elixir leaked | `battle.opponent_elixir_leaked` | [0, ~30] |
| 29 | Avg elixir per play (player) | `total_elixir / max(total_plays, 1)` | [1, 10] |
| 30 | Avg elixir per play (opponent) | same for opponent | [1, 10] |
| 31 | Troop elixir ratio | `troop_elixir / max(total_elixir, 1)` | [0, 1] |
| 32 | Spell elixir ratio | `spell_elixir / max(total_elixir, 1)` | [0, 1] |
| 33 | Building elixir ratio | `building_elixir / max(total_elixir, 1)` | [0, 1] |

**Elixir leak** is the integral of wasted elixir when the bar is full (10/10). It is the strongest single-feature predictor of tilt — core tilt clusters average 12-20 leaked elixir, while dominant-leg games average 2-3.

**Category ratios** (dims 31-33) sum to 1.0 and encode the player's spending strategy: troop-heavy (aggro), spell-heavy (control), or building-heavy (siege/bait).

### 1.3 Tempo Features (8 dimensions)

Tempo captures the rhythm and pacing of play.

| Dim | Feature | Formula | Range |
|-----|---------|---------|-------|
| 34 | Total plays (player) | `team_summary.total_plays` | [4, ~40] |
| 35 | Total plays (opponent) | `opp_summary.total_plays` | [4, ~40] |
| 36 | Play rate (player) | `total_plays / max_tick * 1000` | [0, ~20] |
| 37 | Play rate (opponent) | same for opponent | [0, ~20] |
| 38 | First play timing | `first_play_tick / max_tick` | [0, 1] |
| 39 | Lane split | `right_plays / max(total_team_plays, 1)` | [0, 1] |
| 40 | Average play spacing | `mean(inter_play_gaps) / max_tick` | [0, 1] |
| 41 | Aggression index | `offensive_plays / max(total_team_plays, 1)` | [0, 1] |

**Play rate** is normalized per 1000 game ticks for interpretability. A rate of 10 means ~10 card placements per 1000 ticks (~33 seconds at 30 ticks/sec).

**Lane split** encodes whether plays favor the right lane (>0.5) or left (<0.5). A value near 0.5 indicates balanced dual-lane pressure.

**Aggression index** is the fraction of team plays placed in the opponent's half of the arena (`arena_y > ARENA_Y_MID` where `ARENA_Y_MID = 15750`). This is one of the most discriminative features across the three manifold legs:
- Dominant leg: ~67% aggression
- Contested leg: ~57% aggression
- Overwhelmed leg: ~34% aggression

**Average play spacing** is the mean tick gap between consecutive team plays, normalized by game length. Small values indicate fast cycle speed; large values indicate defensive/reactive play.

### 1.4 Outcome-Adjacent Features (3 dimensions)

These features correlate with game outcome but are not the outcome itself.

| Dim | Feature | Formula | Range |
|-----|---------|---------|-------|
| 42 | Crown differential | `player_crowns - opponent_crowns` | [-3, 3] |
| 43 | Battle duration (normalized) | `battle_duration / 300.0` | [0, ~1.5] |
| 44 | King tower HP (normalized) | `player_king_tower_hp / 10000.0` | [0, 1] |

**Design note:** Including outcome-adjacent features in an embedding used for win/loss analysis creates a subtle circularity. Crown differential and king tower HP are *results* of the game, not strategic inputs. They are included because the primary downstream use is similarity search (finding games that *played out* similarly) rather than win prediction (where they would leak label information). The supervised UMAP target metric ensures the manifold structure correlates with win/loss without these features dominating.

### 1.5 Matchup Context (5 dimensions)

| Dim | Feature | Formula | Range |
|-----|---------|---------|-------|
| 45 | Trophy band | `player_starting_trophies / 10000.0` | [0.5, ~1.3] |
| 46 | Avg elixir cost (player deck) | `mean(card_elixir for player_cards)` | [1, 10] |
| 47 | Avg elixir cost (opponent deck) | `mean(card_elixir for opponent_cards)` | [1, 10] |
| 48 | Evo count (player) | count of cards with `card_variant == "evo"` | [0, 2] |
| 49 | Hero count (player) | count of cards with `card_variant == "hero"` | [0, 2] |

**Trophy band** serves as a skill proxy. A game at 11,000 trophies against a 12,000-trophy opponent encodes a different competitive context than a game between two 7,000-trophy players.

---

## 2. Sequential Feature Vector (17-dim per event)

**Implementation:** `src/tracker/ml/sequence_dataset.py::SequenceDataset`
**Storage:** In-memory PyTorch tensors during training; raw data in `replay_events` table

The TCN operates on the raw event sequence rather than aggregated statistics. Each replay event becomes a 17-dimensional feature vector, paired with an integer card ID that is separately embedded to 16 dimensions by the model's `nn.Embedding` layer.

### 2.1 Per-Event Feature Breakdown

| Index | Feature | Encoding | Dim |
|-------|---------|----------|-----|
| 0 | Side | Binary: 1.0 = team, 0.0 = opponent | 1 |
| 1 | Game tick (normalized) | `min(game_tick / 10000, 1.0)` | 1 |
| 2-5 | Game phase | One-hot: [regular, double, overtime, OT double] | 4 |
| 6 | Arena X (normalized) | `(arena_x - 8750) / 8750` → [-1, 1] | 1 |
| 7 | Arena Y (normalized) | `(arena_y - 15750) / 15750` → [-1, 1] | 1 |
| 8-10 | Lane | One-hot: [left, right, center] | 3 |
| 11 | Play number (normalized) | `min(play_number, 20) / 20` | 1 |
| 12 | Ability used | Binary: 0 or 1 | 1 |
| 13 | Elixir cost (normalized) | `(elixir_cost or 4) / 10.0` | 1 |
| 14-16 | Card type | One-hot: [troop, spell, building] | 3 |

**Total per-event dimension:** 17 (features) + 1 (card_id) = 18 raw values, expanded to 17 + 16 = 33 after card embedding.

### 2.2 Game Phase Boundaries

Game phase is derived from the game tick value using fixed boundaries from the Clash Royale game engine:

| Phase | Tick range | Elixir generation | Behavior |
|-------|-----------|-------------------|----------|
| Regular | [0, 3360) | 1e per 2.8s | Standard play |
| Double elixir | [3360, 5280) | 1e per 1.4s | Accelerated economy |
| Overtime | [5280, 7920) | 1e per 1.4s | Sudden death extension |
| OT double elixir | [7920, ∞) | 1e per 0.7s | Final phase |

### 2.3 Lane Classification

Lane is derived from the `arena_x` coordinate with a center margin:

```python
ARENA_X_MID = 8750
MARGIN = 2000

if arena_x < ARENA_X_MID - MARGIN:    # < 6750
    lane = "left"     → [1, 0, 0]
elif arena_x > ARENA_X_MID + MARGIN:   # > 10750
    lane = "right"    → [0, 1, 0]
else:
    lane = "center"   → [0, 0, 1]
```

### 2.4 Card Type Classification

Card type is looked up from a static dictionary (`CARD_TYPES` in `card_metadata.py`) mapping Title Case card names to `{"troop", "spell", "building"}`. Cards not in the dictionary default to `"troop"`.

### 2.5 Sequence Construction

The `SequenceDataset` constructor:
1. Queries all PvP battles with ≥4 valid (non-`_invalid`) replay events
2. Orders events by `(battle_id, game_tick)` — preserving temporal ordering within each game
3. Encodes each event into a `(card_id, 17-dim feature)` pair
4. Stores each game as a `(card_ids: int64[seq_len], features: float32[seq_len, 17], label: float)` tuple

**Collation:** The `collate_fn` pads variable-length sequences to the maximum length within each batch:
- `card_ids`: zero-padded (index 0 = `<PAD>` token in the embedding layer)
- `features`: zero-padded (all-zero feature vectors for padding positions)
- `lengths`: original sequence lengths for masked pooling

---

## 3. Card Vocabulary

**Implementation:** `src/tracker/ml/card_metadata.py::CardVocabulary`

The vocabulary is built dynamically from the `deck_cards` table at training time, not from a static file. This ensures all cards observed in the corpus are included, even new releases.

### 3.1 Token Structure

| Index | Token | Purpose |
|-------|-------|---------|
| 0 | `<PAD>` | Padding for sequence batching; embedding_idx=0 with `padding_idx=0` |
| 1 | `<UNK>` | Unknown cards (new releases not yet in the training vocabulary) |
| 2+ | Card names | Sorted alphabetically for deterministic ordering |

### 3.2 Name Normalization

Replay events use kebab-case names (`baby-dragon`, `mini-pekka`), while deck cards use Title Case (`Baby Dragon`, `Mini P.E.K.K.A`). The `kebab_to_title()` function handles this conversion with special cases for non-standard capitalization (P.E.K.K.A, X-Bow, The Log).

### 3.3 Vocabulary Growth

When new cards are released between model training runs:
- `embed_new()` detects vocabulary growth (`vocab.size > saved_vocab_size`)
- New cards map to index 0 (`<PAD>`) — their embeddings are effectively random
- Full retraining re-indexes all cards and learns proper embeddings

---

## 4. Feature Matrix Construction

**Implementation:** `src/tracker/ml/features.py::build_feature_matrix()`

### 4.1 Incremental Extraction

The `incremental=True` parameter (default) skips battles that already have a stored feature vector with the current `feature_version`. This enables efficient re-runs:

```python
existing = {battle_id for battle_id in game_features WHERE feature_version = "v1"}
new_battles = [bid for bid in replay_battles if bid not in existing]
```

### 4.2 Batch Processing

Feature vectors are extracted one game at a time (each requires joining 4 tables) and flushed to the database every 500 games. The extraction of 10,000+ games takes ~30 seconds.

### 4.3 Stored Format

Feature vectors are serialized as raw float32 bytes:
```python
def to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def from_blob(data: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32).reshape(dim)
```

A 50-dim vector occupies 200 bytes in SQLite. The `dim` parameter to `from_blob()` accepts -1 for automatic inference from BLOB length.
