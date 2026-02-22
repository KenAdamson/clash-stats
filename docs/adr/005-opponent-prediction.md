# ADR-005: Opponent Play Prediction Model

**Status:** Proposed
**Date:** 2026-02-22
**Depends on:** ADR-001 (Feature Engineering), ADR-003 (Game State Embeddings)

## Context

Clash Royale players at every skill level run patterns. A 2.6 Hog player cycles Ice Golem → Hog Rider at the bridge on a rhythm. A Golem player builds a Golem in the back at 2x elixir almost every game. These patterns exist because human muscle memory and strategic heuristics produce predictable sequences. A model trained on thousands of games can learn these sequences and predict the opponent's next play before it happens.

This isn't theoretical advantage — it's the difference between reactive defense (respond after you see the card) and predictive defense (pre-position your counter because you know what's coming). At 11K+ trophies, reaction time is already optimized. Prediction is the next edge.

## Decision

### 1. Problem Formulation

Given the sequence of all events up to tick T (both sides), predict:

1. **What card** the opponent plays next (classification over ~110 cards, or just the 8 in their known deck)
2. **When** they play it (regression: ticks until next opponent event)
3. **Where** they play it (regression: arena_x, arena_y)

These three predictions combine into a complete forecast: "Opponent will play Hog Rider at the right bridge in approximately 45 ticks."

### 2. Architecture: Autoregressive Sequence Model

```
Input: Interleaved event sequence (both sides) up to tick T
  │
  ▼
Shared Card Embedding (from ADR-003)
  │
  ▼
Side-aware encoding: Concatenate side flag with event features
  │
  ▼
Causal TCN Encoder (from ADR-003/004, shared weights possible)
  │
  ▼
Per-tick hidden states: (seq_len, 256)
  │
  ├──► Card prediction head: Linear(256 → 128) → ReLU → Linear(128 → num_opponent_cards)
  │    Output: Softmax probability distribution over opponent's known cards
  │
  ├──► Timing prediction head: Linear(256 → 64) → ReLU → Linear(64 → 1)
  │    Output: Expected ticks until next opponent play (positive real)
  │
  └──► Position prediction head: Linear(256 → 64) → ReLU → Linear(64 → 2)
       Output: (arena_x, arena_y) normalized coordinates
```

**Constrained card prediction:** Once we've identified the opponent's 8 cards (typically fully scouted by mid-game), the card prediction head can be masked to only output probabilities over those 8 cards. This dramatically improves accuracy by eliminating 100+ impossible options.

**Cycle tracking as input feature:** The opponent's card cycle is partially observable. If we've seen them play cards A, B, C, D since their last play of card E, then E is either in hand or next in queue. An explicit cycle position counter per opponent card feeds into the model as an additional input feature.

### 3. Training

**Labels:** For each event in the sequence where `side == "opponent"`, the label is:
- Card: the card_name of that event
- Timing: game_tick of that event minus game_tick of the previous event
- Position: (arena_x, arena_y) of that event

**Loss function:**
- Card: Cross-entropy (weighted by card frequency to handle imbalanced decks)
- Timing: Smooth L1 loss (robust to outliers from long pauses)
- Position: MSE on normalized coordinates
- Total: Weighted sum with tunable coefficients

**Teacher forcing during training:** The model sees the actual previous events during training (not its own predictions). At inference time, we only predict the *next* opponent play, not a full future sequence.

### 4. Opponent Modeling by Archetype

Rather than training a single model for all opponents, we can condition on archetype:

**Option A: Archetype-conditioned single model**
- Add archetype label as an input feature (one-hot or embedding)
- Single model learns archetype-specific patterns
- More data-efficient (all games contribute to shared weights)

**Option B: Per-archetype specialist models**
- Train separate small models for each archetype (2.6 Hog, Golem beatdown, Log bait, etc.)
- Higher accuracy per archetype but needs more data per archetype
- Risk of overfitting with small per-archetype sample sizes

**Decision: Option A** (archetype-conditioned single model) until per-archetype sample sizes exceed 500 games each, then evaluate specialist models.

### 5. Cycle Prediction vs Play Prediction

Two distinct prediction tasks:

**Play prediction** (what, when, where) is the immediate tactical question — what does the opponent do next?

**Cycle prediction** is the strategic question — what is the opponent's overall card rotation pattern? This is modeled as a Markov chain over the 8-card deck:

```
P(next_card | last_4_cards_played)
```

The transition matrix is estimated directly from replay data. With 8 cards, this is a tractable 8×8 matrix (or 8^4 × 8 for 4th-order Markov, which is sparse but estimable with sufficient data).

**Cycle prediction doesn't need a neural model** — it's a counting problem. But it feeds into the neural play prediction model as an informative prior: the model knows what's *likely* in the cycle and refines with contextual information (game state, elixir, timing).

### 6. Elixir State Inference

The opponent's elixir is hidden information, but it's partially inferable:

```
opponent_elixir(T) ≈ 5 + generation_rate * T - sum(costs of known opponent plays before T)
```

This is approximate because:
- We don't know when the opponent reaches 10 and starts leaking
- Mirror costs vary
- Pump generation is opponent-side (not common at 11K but exists)

The inferred opponent elixir state is a critical input to the timing prediction head: if the opponent is at 2 elixir, they can't play a 5-elixir card for ~8 more ticks. The model should learn this constraint from data, but providing the explicit elixir estimate as an input feature accelerates learning.

### 7. Evaluation Metrics

**Card prediction:**
- Top-1 accuracy: did we predict the exact card?
- Top-3 accuracy: was the actual card in our top 3 predictions?
- Accuracy by game phase (early game predictions are harder — more uncertainty)
- Accuracy by number of cards scouted (improves as opponent's deck is revealed)

**Timing prediction:**
- Mean absolute error in ticks
- Percentage of predictions within ±30 ticks (~1 second) of actual

**Position prediction:**
- Mean Euclidean distance between predicted and actual placement
- Lane accuracy: did we predict the correct lane (left/right/center)?

**Practical metric: defensive preparation time.** If the model predicts "Hog Rider at right bridge in 45 ticks" and the actual play is Hog Rider at right bridge in 50 ticks, the player gets 50 ticks of preparation time instead of the ~15 ticks of human reaction time. That's a 3x improvement in response window.

### 8. Inference Integration

The prediction model runs as a post-game analysis tool (not real-time during gameplay — that would require screen reading and is outside scope). But the *insights* from prediction inform live play:

**Pattern reports:** "Against 2.6 Hog, after opponent plays Ice Golem at the bridge, Hog Rider follows within 2 seconds 83% of the time."

**Counter-timing maps:** "Against Golem decks, Golem is played in the back left 67% of the time at 2x elixir. When Golem is played back left, Night Witch follows 72% of the time within 3 seconds."

**Surprise index:** How predictable is each opponent archetype? High entropy in the prediction distribution = unpredictable. Low entropy = telegraphed. This tells you which matchups reward prediction (low entropy) vs which require reactive play (high entropy).

### 9. Implementation

```
src/tracker/
├── ml/
│   ├── opponent_model.py     ← Architecture, training, inference
│   ├── cycle_tracker.py      ← Markov chain cycle prediction
│   └── pattern_report.py     ← Human-readable pattern extraction
```

CLI additions:
```
clash-stats --train-opponent            # Train opponent prediction model
clash-stats --predict BATTLE_ID         # Show predictions vs actuals for a game
clash-stats --patterns ARCHETYPE        # Show common play patterns for an archetype
clash-stats --cycle-matrix ARCHETYPE    # Show card cycle transition matrix
clash-stats --surprise                  # Predictability ranking by archetype
```

## Consequences

### Positive
- Quantifies opponent predictability — transforms intuition ("I know what Hog players do") into statistics
- Pattern reports are immediately useful without any ML — pure counting from replay data
- Cycle tracking via Markov chains works with 50+ games per archetype
- Defensive preparation time improvement is a concrete, measurable competitive advantage
- Surprise index identifies which matchups benefit most from study vs which are inherently chaotic

### Negative
- Play prediction accuracy is fundamentally bounded — opponents don't follow patterns perfectly, especially at higher trophy ranges where players adapt mid-game
- Real-time inference during gameplay is out of scope (would require computer vision pipeline)
- Card prediction before full scouting is nearly random — model is only useful from mid-game onward
- Position prediction is the weakest component — placement is highly context-dependent

### Scale Requirements

| Games per archetype | Capability |
|--------------------|------------|
| 20 | Markov chain cycle estimation (noisy) |
| 50 | Reliable cycle matrices, basic pattern reports |
| 200 | Neural play prediction with ~40% top-1 card accuracy |
| 500 | ~55% top-1 card accuracy, meaningful timing predictions |
| 2,000+ | Position prediction becomes informative, per-player modeling possible for frequent opponents |
