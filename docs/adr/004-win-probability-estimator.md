# ADR-004: Real-Time Win Probability Estimator

**Status:** Proposed
**Date:** 2026-02-22
**Depends on:** ADR-001 (Feature Engineering), ADR-003 (Game State Embeddings)

## Context

In baseball, Win Probability Added (WPA) quantifies the impact of every play on the game's outcome. A home run in the 9th inning of a tie game has a different WPA than a home run in a 10-0 blowout. Clash Royale has no equivalent metric. Every replay review is subjective — "I think I lost when I played Pekka into Inferno Dragon" — without any way to quantify *how much* that play actually cost.

This model produces P(win) at every game tick, so that every play has a measurable impact on the outcome. The delta in P(win) before and after a play is that play's WPA — the objective cost of a mistake or value of a good read.

## Decision

### 1. Model Architecture

The win probability estimator is a causal variant of the TCN from ADR-003. "Causal" means it only looks backward — P(win) at tick T is computed from events at ticks ≤ T only. No future information leaks into the prediction.

```
Input: Event sequence up to tick T (from ADR-001 features)
  │
  ▼
Shared Card Embedding (from ADR-003, frozen or fine-tuned)
  │
  ▼
Causal TCN Encoder (same architecture as ADR-003, but with causal masking)
  │
  ▼
Per-tick hidden states: (seq_len, 256)
  │
  ▼
Win probability head: Linear(256 → 64) → ReLU → Linear(64 → 1) → Sigmoid
  │
  ▼
Output: P(win) at each tick in [0, 1]
```

**Why causal masking matters:** Without it, the model could "cheat" by looking at future events to predict the outcome. With causal masking, each prediction uses only the information available at that point in the game — exactly what a player would know during live play.

### 2. Training Strategy

**Labels:** Every tick in every game gets the same label — the final game result (1 = win, 0 = loss). The model learns to predict this outcome from partial information, and the uncertainty of the prediction naturally decreases as more of the game is revealed.

**Loss function:** Binary cross-entropy, averaged over all ticks in the sequence. Early ticks contribute more uncertainty (closer to 0.5), late ticks contribute more confident predictions (closer to 0 or 1).

**Calibration:** After training, apply Platt scaling or isotonic regression to ensure that "P(win) = 0.7" means the player actually wins 70% of the time across all games where the model outputs 0.7 at that tick. Calibration is critical — an uncalibrated model's probabilities are meaningless for WPA computation.

**Class weighting:** If the training set is imbalanced (e.g., 62% win rate), apply inverse frequency weighting so the model doesn't just predict "win" for everything.

### 3. Win Probability Added (WPA)

For each event at tick T:

```
WPA(event_T) = P(win | events_1..T) - P(win | events_1..T-1)
```

Positive WPA = this play improved your win probability.
Negative WPA = this play hurt your win probability.

**Aggregate WPA per card:** Sum the WPA of all deployments of a given card across a game. "Pekka had a cumulative WPA of -0.15 this game" means Pekka deployments collectively reduced your win probability by 15 percentage points.

**Aggregate WPA per game phase:** Sum WPA for all events in regular time vs double elixir vs overtime. "Your double elixir play had a WPA of +0.30" means you gained 30 percentage points of win probability during double elixir — that's where you won the game.

### 4. Decision Point Analysis

The most valuable output isn't P(win) itself — it's the **derivative**. Points where P(win) changes sharply are the critical decision points.

```
criticality(T) = |P(win | events_1..T) - P(win | events_1..T-1)|
```

Events with `criticality > 0.10` are "game-defining plays" — moments where the outcome swung by 10+ percentage points on a single card placement. These are the plays worth reviewing in detail.

**Automatic highlight extraction:** Given a replay, identify the top-5 events by criticality. These are the 5 moments that most determined the game's outcome. This replaces subjective "I think I lost when..." with quantitative "the game was decided at tick 1847 when opponent played Inferno Dragon (+0.22 WPA for opponent)."

### 5. Pre-Game Win Probability

Before any cards are played, the model should output a starting P(win) based on:
- Deck matchup (your 8 cards vs their 8 cards)
- Trophy differential
- Structural features (average elixir cost ratio, splash coverage, win condition type)

This is the "pre-game odds" — how favorable is this matchup before any gameplay happens? The difference between pre-game P(win) and post-game result measures *outperformance*. If pre-game odds were 35% and you won, you beat the matchup. If pre-game odds were 80% and you lost, something went wrong in execution.

**Implementation:** A separate lightweight MLP that takes per-game context features (from ADR-001) and outputs P(win). This can be trained on the full battle history (not just games with replay data) since it only needs deck composition and result.

### 6. Volatility Index

Some games have a P(win) curve that's nearly flat (dominant wins or hopeless losses). Others oscillate wildly. The **volatility** of the P(win) curve characterizes how "swingy" a game is:

```
volatility = std(diff(P(win))) over all ticks
```

High volatility = back-and-forth game, outcome determined by a few key plays.
Low volatility = one-sided game, outcome was determined by deck matchup.

Cross-reference with matchup: if all games against archetype X have high volatility, that's a coinflip matchup. If they have low volatility and you lose, it's a hard counter. This is a more nuanced signal than raw win rate.

### 7. Output Schema

```sql
CREATE TABLE win_probability (
    id INTEGER PRIMARY KEY,
    battle_id TEXT REFERENCES battles(battle_id),
    game_tick INTEGER NOT NULL,
    win_prob REAL NOT NULL,
    wpa REAL,                          -- Delta from previous tick
    criticality REAL,                  -- Absolute WPA
    event_index INTEGER,               -- Which replay event caused this (nullable for interpolated ticks)
    model_version TEXT NOT NULL,
    UNIQUE(battle_id, game_tick, model_version)
);
CREATE INDEX idx_wp_battle ON win_probability(battle_id);
CREATE INDEX idx_wp_criticality ON win_probability(criticality);

CREATE TABLE game_wp_summary (
    battle_id TEXT PRIMARY KEY REFERENCES battles(battle_id),
    pre_game_wp REAL,                  -- P(win) before any plays
    final_wp REAL,                     -- P(win) at last tick (should be ~0 or ~1)
    max_wp REAL,                       -- Highest P(win) during game
    min_wp REAL,                       -- Lowest P(win) during game
    volatility REAL,                   -- Std of P(win) differences
    top_positive_wpa_card TEXT,        -- Card with highest cumulative positive WPA
    top_negative_wpa_card TEXT,        -- Card with highest cumulative negative WPA
    critical_tick INTEGER,             -- Tick with highest criticality
    critical_card TEXT,                -- Card played at critical tick
    model_version TEXT NOT NULL
);
```

### 8. Visualization

The P(win) curve is the centerpiece visualization for replay analysis:

```
P(win)
1.0 ─┐
     │          ╭──╮    Pekka connects
0.8 ─┤     ╭───╯  │
     │  ╭──╯      │
0.6 ─┤──╯         │
     │             │    Inferno Dragon melts Pekka
0.4 ─┤             ╰──╮
     │                 ╰───╮
0.2 ─┤                     ╰──── Loss
     │
0.0 ─┴──────────────────────────────
     0:00  0:30  1:00  1:30  2:00  2:30
```

Dashboard integration: render this as a Chart.js line chart on the web dashboard, with hover annotations showing the card played at each inflection point.

### 9. Implementation

```
src/tracker/
├── ml/
│   ├── win_probability.py    ← Model definition, training, inference
│   ├── wpa.py                ← WPA computation, critical play extraction
│   └── calibration.py        ← Platt scaling, calibration diagnostics
```

CLI additions:
```
clash-stats --train-wp                     # Train win probability model
clash-stats --wp BATTLE_ID                 # Show P(win) curve for a game
clash-stats --wp-critical BATTLE_ID        # Show top-5 critical plays
clash-stats --wp-card-impact               # Aggregate WPA by card across all games
clash-stats --wp-pregame                   # Pre-game odds for recent matchups
```

## Consequences

### Positive
- Transforms subjective replay analysis into quantitative decision evaluation
- WPA per card is the definitive answer to "which card is carrying/hurting my deck?"
- Pre-game win probability enables honest matchup assessment independent of recency bias
- Critical play extraction automates the most valuable part of replay review — finding the moments that matter
- Volatility index distinguishes coinflip matchups from hard counters, which raw win rate cannot

### Negative
- Causal masking reduces model accuracy compared to full-sequence models (can't see future context)
- Calibration requires a held-out validation set, which is expensive when data is scarce
- WPA assumes the model's probability estimates are correct — garbage model = garbage WPA
- Per-tick storage is ~100-200 rows per game, which scales to millions of rows over thousands of games (manageable in SQLite but worth monitoring)

### Scale Requirements

| Games with replay data | Capability |
|----------------------|------------|
| 200 | Pre-game win probability only (MLP on deck features) |
| 500 | Causal TCN with supervised training, ~70% calibrated accuracy |
| 2,000 | Reliable WPA computation, meaningful critical play identification |
| 5,000+ | Per-archetype win probability models, fine-grained card impact analysis |
