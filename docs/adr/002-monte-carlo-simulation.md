# ADR-002: Monte Carlo Simulation Framework

**Status:** Implemented
**Date:** 2026-02-22
**Implemented:** 2026-02-23
**Depends on:** ADR-001 (Feature Engineering)

## Context

Monte Carlo methods require no training data minimum, no GPU, and no hyperparameter tuning. They run on day one with whatever replay data exists. This makes them the right first analytical tool — they produce actionable insights from 50 games while the neural models wait for scale.

The replay data provides the empirical distributions that Monte Carlo samples from. Instead of hand-coding "Pekka does X damage to Inferno Dragon," we sample from *observed* Pekka-vs-Inferno-Dragon interactions across real games.

## Decision

### 1. Opening Hand Simulator

**Problem:** An 8-card deck draws 4 cards as the opening hand plus 1 in queue. Starting hand quality varies dramatically — Pekka + Graveyard + Witch + Executioner (19 elixir, nothing playable for 10+ seconds) vs Bats + Miner + Goblin Curse + Arrows (10 elixir, immediate tempo plays). What's the actual probability distribution of opening hand quality, and how does it correlate with win rate?

**Method:**
1. Enumerate all C(8,4) = 70 possible opening hands
2. For each hand, compute: total elixir cost, cheapest playable card, time-to-first-play (cheapest cost / generation rate), number of sub-3-elixir cards
3. Cross-reference with replay data: for games where the first team event matches a given opening card, what's the observed win rate?
4. Run 100K simulated draws, weight by observed outcomes

**Output:**
- Distribution of opening hand "quality scores" (composite of cost, tempo, flexibility)
- Expected win rate by opening hand composition
- Per-matchup opening hand recommendations: "Against Hog cycle, pray for Bats + Arrows in opener"

**Scale requirement:** Works with 50+ games. Improves with more data (tighter confidence intervals on per-hand win rates).

### 2. Elixir Economy Simulator

**Problem:** The elixir trade is the fundamental unit of Clash Royale strategy. Every card interaction is an elixir exchange. Across an entire game, the player who gets more value per elixir spent wins. But elixir efficiency isn't a single number — it's a time-varying stochastic process that depends on the sequence of interactions.

**Method:**
1. From replay data, build an **interaction matrix**: for each pair (your_card, opponent_card), compute the distribution of elixir outcomes. "When I play Pekka (7e) and opponent responds with Inferno Dragon (4e), the observed tower damage distribution is X and Pekka survival rate is Y%."
2. Model a game as a sequence of elixir exchanges sampled from these distributions
3. At each timestep:
   - Sample available cards from hand (cycle position modeled as a Markov chain)
   - Sample opponent's play from observed frequency distribution for their archetype at this game phase
   - Look up the interaction outcome distribution
   - Update elixir state for both sides
4. Run 10K simulated games per matchup

**Output:**
- Expected elixir differential over time by matchup
- Variance of elixir differential (high variance = coinflip matchup, low variance = deterministic)
- Identification of "break points" — game ticks where elixir differential swings most violently
- Elixir leak projections: given your deck's average cost vs opponent's cycle speed, what's the expected leak in double elixir?

**Key distributions to estimate from data:**
- `P(opponent_response | your_card, archetype, game_phase)` — What does the opponent play after you play X?
- `damage(your_card, opponent_response)` — Tower damage resulting from this exchange
- `net_elixir(your_card, opponent_response)` — Elixir differential after the exchange resolves

**Scale requirement:** 100+ games for rough estimates, 500+ for stable per-matchup distributions. Below 100 games, pool across similar archetypes.

### 3. Matchup Win Probability

**Problem:** The API data gives win/loss records per opponent card, but sample sizes are small (e.g., 1W/3L vs Inferno Dragon in the current window). Monte Carlo can estimate confidence intervals and project forward.

**Method:**
1. Model win rate against each archetype as a Beta distribution: `Beta(wins + 1, losses + 1)`
2. Sample from the posterior to get credible intervals
3. For new/unseen matchups, use a hierarchical prior informed by structural similarity (deck elixir cost differential, splash vs single-target composition, etc.)
4. Combine with elixir economy simulations from §2 to generate forward-looking win probability estimates

**Output:**
- Per-matchup win probability with 95% credible intervals
- "True" win rate estimates that account for small sample sizes (Bayesian shrinkage toward prior)
- Matchup volatility: how much does win rate vary between simulated games? (A 50% win rate with low variance is different from 50% with high variance)

**Scale requirement:** Works immediately with Bayesian priors. Improves continuously as data accumulates.

### 4. Card Substitution Analysis

**Problem:** You can't test deck changes on ladder without risking trophies. But you can simulate: "If I replace Arrows (3e, kills swarms, no stun) with Zap (2e, doesn't kill Goblins, has stun), how does my elixir economy change across matchups?"

**Method:**
1. Take the interaction matrix from §2
2. Replace the row/column for the substituted card with the new card's estimated interactions
3. New card interactions are estimated from:
   - Card stats (damage, HP, cost, targets, speed) mapped to expected outcomes via a simple rule-based model
   - Observed interactions of the new card in opponent decks (if the new card appears in opponent decks in your replay data, you have the other side of those interactions)
   - Community win rate data from RoyaleAPI for the new card in similar deck archetypes (optional, external data source)
4. Re-run the elixir economy simulation with the modified deck
5. Compare distributions: does the substitution improve expected value? Reduce variance? Shift specific matchups?

**Output:**
- Per-matchup expected win rate delta from the substitution
- Elixir efficiency change
- Risk assessment: does the substitution create any new hard-counter matchups?
- Confidence level: how much of the estimate is observed data vs imputed from card stats?

**Scale requirement:** This is the most data-hungry simulation because it requires dense interaction matrices. Below 200 games, the card-vs-card interaction estimates have wide confidence intervals. At 500+ games, the estimates stabilize. The top-ladder corpus (ADR-007) dramatically accelerates this by providing interaction observations from thousands of games.

### 5. Implementation

```
src/tracker/
├── simulation/
│   ├── __init__.py
│   ├── opening_hand.py    ← §1: Hand quality distributions
│   ├── elixir_economy.py  ← §2: Elixir exchange simulation
│   ├── matchup_model.py   ← §3: Bayesian matchup estimation
│   ├── card_substitution.py ← §4: Deck modification analysis
│   └── interaction_matrix.py ← Shared: card-vs-card outcome distributions
```

CLI additions:
```
clash-stats --sim-hands              # Opening hand analysis
clash-stats --sim-elixir ARCHETYPE   # Elixir economy vs archetype
clash-stats --sim-matchups           # Full matchup probability table
clash-stats --sim-swap OLD NEW       # Card substitution analysis
```

All simulation outputs are stored in a `simulation_results` table for querying and comparison over time.

## Consequences

### Positive
- Runs immediately with current data volume (200+ games)
- No GPU or training infrastructure required
- Results are interpretable — distributions and confidence intervals, not black-box predictions
- Card substitution analysis is directly actionable for deck evolution decisions
- Bayesian matchup estimation is strictly better than raw win/loss counts for small samples

### Negative
- Interaction matrix is sparse for rare card combinations — some matchups will have zero observed interactions
- Card substitution estimates for unobserved cards rely heavily on rule-based imputation, which may not capture emergent interactions (e.g., synergies that only appear in practice)
- Simulation assumes independence between consecutive exchanges, which is wrong — a Pekka surviving one exchange changes the state for the next. Addressed partially by the sequence model in ADR-005.

### Scale Thresholds

| Games in corpus | Capability unlocked |
|-----------------|-------------------|
| 50 | Opening hand analysis, Bayesian matchup estimation with wide priors |
| 200 | Elixir economy simulation with pooled archetype distributions |
| 500 | Per-archetype interaction matrices with reasonable confidence |
| 2,000 | Dense card-vs-card interaction matrices, card substitution analysis |
| 10,000+ | Per-trophy-band modeling, meta-shift detection over time (requires ADR-007) |
