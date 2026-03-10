# Win Probability Estimator — Architecture & Runbook

**Version:** 1.0
**Date:** 2026-03-10
**Implements:** ADR-004

## 1. What This Does

Produces P(win) at every game tick, so every card placement has a measurable impact. The delta in P(win) before and after a play is that play's **Win Probability Added (WPA)** — the objective cost of a mistake or value of a good read.

```
WPA(event_T) = P(win | events_1..T) - P(win | events_1..T-1)
```

- **Positive WPA** = this play improved your win probability
- **Negative WPA** = this play hurt your win probability
- **Criticality** = |WPA| — how much the game swung on a single play

## 2. Model Architecture

**Implementation:** `src/tracker/ml/win_probability.py::WinProbabilityModel`

```
Input:
  card_ids:  (batch, seq_len)       int64
  features:  (batch, seq_len, 17)   float32
  lengths:   (batch,)               int64

  → Card Embedding: nn.Embedding(vocab_size, 16, padding_idx=0)
  → Concatenate → (batch, seq_len, 33)
  → Transpose → (batch, 33, seq_len)

  → TCN Encoder (reused from ADR-003):
    6 TemporalBlocks with dilated causal convolutions
    channels: [33→64, 64→64, 64→128, 128→128, 128→256, 256→256]
    dilations: [1, 2, 4, 8, 16, 32]

  → Per-tick classification head:
    Conv1d(256→64, kernel=1) → ReLU → Dropout → Conv1d(64→1, kernel=1)

Output:
  logits: (batch, seq_len)    → sigmoid → P(win) ∈ [0, 1]
```

**Key difference from ADR-003 TCN:** The embedding model uses global pooling (mean + max + last) to produce a single 128-dim vector per game. The WP model skips pooling entirely — each tick gets its own prediction. The TCN's causal convolutions already ensure each output depends only on past events, making this naturally causal without additional masking.

## 3. Transfer Learning

When a trained ADR-003 TCN exists (`data/ml_models/tcn_v1.pt`), the WP model loads its card embedding and TCN encoder weights. By default, these are **frozen** — only the per-tick head trains. This:

- Prevents catastrophic forgetting of learned game representations
- Reduces trainable parameters dramatically (~17K head-only vs ~2M full model)
- Converges faster with less data

The transfer happens via `WinProbabilityModel.from_pretrained_tcn()`.

## 4. Training Details

**Implementation:** `src/tracker/ml/wp_training.py::WPTrainer`

- **Labels:** Every tick in a game gets the same label — the final result (1.0 = win, 0.0 = loss). The model learns that early ticks should predict ~0.5 (uncertain) and late ticks should predict ~0.0 or ~1.0 (confident).
- **Loss:** Binary cross-entropy per tick, masked to exclude padding, averaged over real ticks only.
- **Class weighting:** Inverse frequency weighting (`pos_weight = n_losses / n_wins`) handles imbalanced win rates.
- **Optimizer:** AdamW, lr=5e-4, weight_decay=1e-4
- **Scheduler:** CosineAnnealingLR over 50 epochs
- **Early stopping:** Patience 10 on validation loss
- **Validation metric:** Last-tick accuracy (final P(win) prediction vs actual result)
- **Train/val split:** 80/20, deterministic (last 20% by battle_time = most recent games as validation)

## 5. Storage Schema

**Tables** (migration `010_add_win_probability.py`):

```sql
-- Per-tick win probability
win_probability (
    id              INTEGER PRIMARY KEY,
    battle_id       TEXT REFERENCES battles(battle_id),
    game_tick       INTEGER NOT NULL,
    win_prob        REAL NOT NULL,      -- P(win) at this tick
    wpa             REAL,               -- Delta from previous tick
    criticality     REAL,               -- |WPA|
    event_index     INTEGER,            -- Index into replay event sequence
    model_version   TEXT NOT NULL,
    UNIQUE(battle_id, game_tick, model_version)
)

-- Per-game summary
game_wp_summary (
    battle_id               TEXT PRIMARY KEY REFERENCES battles(battle_id),
    pre_game_wp             REAL,       -- P(win) at first event
    final_wp                REAL,       -- P(win) at last event
    max_wp                  REAL,       -- Peak during game
    min_wp                  REAL,       -- Trough during game
    volatility              REAL,       -- std(WPA) — how swingy the game was
    top_positive_wpa_card   TEXT,       -- Card with highest cumulative positive WPA
    top_negative_wpa_card   TEXT,       -- Card with highest cumulative negative WPA
    critical_tick           INTEGER,    -- Tick with highest criticality
    critical_card           TEXT,       -- Card played at critical tick
    model_version           TEXT NOT NULL
)
```

**ORM models:** `src/tracker/ml/wp_storage.py`

**Scale note:** ~100-200 rows per game in `win_probability`. At 50K games, that's 5-10M rows — well within SQLite's capability, but keep an eye on query performance. The `idx_wp_battle` index covers the common access pattern (all ticks for one game).

## 6. Runbook

### Prerequisites

- Replay events in the database (`replay_events` table with valid card plays)
- Minimum 50 games with ≥4 replay events each
- (Optional) Trained ADR-003 TCN at `data/ml_models/tcn_v1.pt` for transfer learning

### Train the model

```bash
# From host via docker exec
docker exec cr-tracker clash-stats --train-wp

# Or directly
clash-stats --train-wp --db data/clash_royale_history.db
```

**What happens:**
1. Loads all eligible PvP games with replay events
2. Computes class weight for win/loss imbalance
3. Checks for ADR-003 TCN checkpoint — if found, transfers card embedding + encoder weights and freezes them
4. Trains per-tick classification head with masked BCE loss
5. Early stopping on validation loss (patience=10)
6. Loads best checkpoint
7. Clears any previous WP data for this model version
8. Runs inference on all games: stores per-tick P(win), WPA, criticality
9. Computes and stores per-game summaries (volatility, top cards, critical tick)

**Output:** Checkpoint saved to `data/ml_models/wp_v1.pt`

**Expected training time:** ~5-15 minutes on CPU depending on game count. Faster on XPU/CUDA.

### View a game's P(win) curve

```bash
clash-stats --wp <BATTLE_ID> --db data/clash_royale_history.db
```

Prints an ASCII chart of P(win) over the game timeline with the 50% reference line and top-5 biggest swings annotated.

### Find critical plays

```bash
clash-stats --wp-critical <BATTLE_ID> --db data/clash_royale_history.db
```

Shows the top-10 highest-criticality events — the plays that most determined the game's outcome. A criticality of 0.10+ means the game swung by 10+ percentage points on a single card placement.

### Query summaries

```bash
# Swingiest games (most volatile P(win) curves)
csq "SELECT battle_id, volatility, critical_card,
     top_positive_wpa_card, top_negative_wpa_card
     FROM game_wp_summary ORDER BY volatility DESC LIMIT 10"

# Games where you started behind but won (outperformance)
csq "SELECT battle_id, pre_game_wp, final_wp
     FROM game_wp_summary
     WHERE pre_game_wp < 0.4 AND final_wp > 0.5
     ORDER BY pre_game_wp ASC LIMIT 10"

# Card that costs you the most games
csq "SELECT top_negative_wpa_card, COUNT(*) as games
     FROM game_wp_summary
     GROUP BY top_negative_wpa_card
     ORDER BY games DESC LIMIT 10"

# Card that carries you the most
csq "SELECT top_positive_wpa_card, COUNT(*) as games
     FROM game_wp_summary
     GROUP BY top_positive_wpa_card
     ORDER BY games DESC LIMIT 10"
```

### Retrain after new data

When new games accumulate, retrain from scratch:

```bash
clash-stats --train-wp --db data/clash_royale_history.db
```

This clears previous WP data for the model version before storing new results, so it's idempotent.

## 7. Interpreting Results

### WPA values

| WPA Range | Interpretation |
|-----------|---------------|
| ±0.01-0.03 | Routine play — minimal impact |
| ±0.03-0.10 | Meaningful swing — good/bad trade |
| ±0.10-0.20 | Game-defining play — review this |
| ±0.20+ | Catastrophic mistake or brilliant read |

### Volatility

| Volatility | Interpretation |
|-----------|---------------|
| < 0.02 | One-sided game — outcome decided by deck matchup |
| 0.02-0.05 | Normal game — some back-and-forth |
| 0.05-0.10 | Swingy game — multiple momentum shifts |
| > 0.10 | Coinflip — outcome came down to 1-2 key plays |

### Pre-game WP

The P(win) at the first event approximates pre-game odds. Cross-reference with `game_wp_summary.final_wp`:

- **pre_game_wp < 0.4, won** → You outplayed a bad matchup
- **pre_game_wp > 0.6, lost** → Execution error in a favorable matchup
- **pre_game_wp ≈ 0.5** → Even matchup, decided by gameplay

## 8. File Map

```
src/tracker/
├── ml/
│   ├── win_probability.py     ← WinProbabilityModel (causal TCN + per-tick head)
│   ├── wp_dataset.py          ← wp_collate_fn (per-tick labels + mask)
│   ├── wp_training.py         ← WPTrainer, train_wp() pipeline
│   └── wp_storage.py          ← WinProbability, GameWPSummary ORM models
├── alembic/versions/
│   └── 010_add_win_probability.py  ← Schema migration
├── cli.py                     ← --train-wp, --wp, --wp-critical args
└── reporting.py               ← print_wp_curve(), print_wp_critical()
```

## 9. Future Work (ADR-004 Phase 2+)

- **Platt scaling / isotonic regression** for probability calibration (`ml/calibration.py`)
- **Pre-game MLP** trained on full battle history (not just replay games) for deck-level matchup odds
- **Dashboard integration** — Chart.js line chart with hover annotations for card plays at inflection points
- **Per-archetype WP models** — fine-tuned models for specific matchup types
- **Aggregate WPA per card** — "which card is carrying/hurting your deck across all games?"
