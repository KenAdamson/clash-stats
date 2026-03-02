# Empirical Findings from Manifold Analysis

**Version:** 1.0
**Date:** 2026-03-01
**Corpus:** ~10,600 games (personal + top-ladder), ~3,800 with replay event data

## 1. Three-Leg Manifold Structure

### 1.1 Discovery

When 128-dimensional TCN game embeddings are projected to 3D via UMAP, the resulting point cloud exhibits a consistent three-armed structure. K-means ($k=3$) on the 3D coordinates separates the three legs cleanly. The structure persists across retraining runs and is robust to hyperparameter variation in both the TCN and UMAP stages.

### 1.2 Leg Characterization (10,613 games)

| Metric | Leg 0: Dominant | Leg 1: Contested | Leg 2: Overwhelmed |
|--------|----------------|-------------------|---------------------|
| Game count | 3,013 | 5,077 | 2,523 |
| Win rate | 62.5% | 53.5% | 50.5% |
| Aggression index | 67.2% | 57.1% | 34.1% |
| Avg plays/game | 26 | 25 | 24 |
| Avg player leak | 2.3e | 3.1e | 4.1e |
| Avg crown diff | +1.2 | +0.3 | -0.1 |
| Alternation rate | 55.3% | 54.1% | 52.8% |

### 1.3 Interpretation

The three legs represent three qualitatively distinct game states, distinguished primarily by **aggression** (spatial control of the arena) and **elixir discipline** (leak rate):

**Dominant (Leg 0):** The player controls the game. 67% of card placements are in the opponent's half of the arena. Elixir management is tight (2.3e leaked avg). Crown differential is strongly positive. This leg represents games where the player's strategy is executing as designed — sustained offensive pressure with efficient resource usage.

**Contested (Leg 1):** The largest segment. Games are competitive with near-equal aggression (~57%). Win rate is slightly above 50%, suggesting the player has a marginal edge in contested situations. These are the "coin flip" games where execution quality determines the outcome.

**Overwhelmed (Leg 2):** The player is pushed into their own half (34% aggression — meaning 66% of their plays are defensive). Elixir leak doubles compared to the dominant leg. Despite being on the back foot, win rate is still 50.5% — suggesting the player's deck/skill has enough resilience to win even from a defensive posture. However, the experience is qualitatively different: narrow 0-1 or 1-2 losses dominate.

### 1.4 Aggression as the Primary Discriminant

The aggression index (fraction of team plays in the opponent's arena half) shows the widest spread across legs:

```
Dominant:     ████████████████████████████████████████░░░░░░░░░░░░░  67%
Contested:    ██████████████████████████████████░░░░░░░░░░░░░░░░░░░  57%
Overwhelmed:  ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  34%
```

This 33-percentage-point gap between dominant and overwhelmed is the largest single-feature separation in the manifold. It reflects the fundamental dichotomy: when you control territory, you win; when you don't, you struggle.

### 1.5 Elixir Leak Gradient

Elixir leak increases monotonically from dominant to overwhelmed:

```
Dominant:     ██░░░░░░░░  2.3e
Contested:    ███░░░░░░░  3.1e
Overwhelmed:  ████░░░░░░  4.1e
```

This is consistent with the tilt hypothesis: being overwhelmed leads to suboptimal play, which leads to elixir waste, which compounds the problem. The tilt detector's `LEAK_ELEVATED` threshold of 6.0 is set approximately 1.5x above the overwhelmed-leg average, ensuring it triggers only for genuinely degraded play, not merely being on the defensive.

---

## 2. High-Z Outlier Cluster

### 2.1 Discovery

A distinct cluster of 112 games appears at UMAP-3 coordinate values of 30-40, well above the main body of the manifold (which occupies Z ≈ -20 to +20). HDBSCAN assigns all 112 to cluster 58.

### 2.2 Profile

| Metric | High-Z Cluster | Main Body |
|--------|----------------|-----------|
| Game count | 112 | ~10,500 |
| Personal games | 0 | ~2,800 |
| Score pattern | 79% single-crown (1-0/0-1) | ~45% single-crown |
| Overtime rate | 58% past tick 5000 | ~25% |
| Avg total plays | 40.7 | 25.4 |
| Median player trophies | 12,500 | ~9,500 |
| % above 12,000 trophies | 70% | ~15% |
| Median max tick | 5,332 | 3,563 |

### 2.3 Interpretation

These are **elite stalemates** — games between the very best players in the world who are so evenly matched that neither can achieve a decisive advantage. The signature is:

1. **No personal games:** The user (at 10,900-11,100 trophies) doesn't appear in this cluster. The games are exclusively between 12,000+ trophy players, a skill tier above the personal data range.

2. **Extreme game length:** 58% extend past overtime. The median max tick is 50% higher than the main body. These games go the distance because neither player can break through.

3. **High play density:** 40+ plays per game vs 25 average. Both sides are deploying cards at maximum efficiency with minimal leak — there's no wasted elixir because both players are near-optimal.

4. **Single-crown resolution:** 79% end with a score of 1-0 or 0-1. The winning margin is a single tower, often in overtime. Three-crowns are essentially nonexistent at this level.

5. **Topological isolation:** The cluster occupies a distinct region of the UMAP manifold because the event sequence structure (long, dense, even) is qualitatively different from any pattern in the main body.

### 2.4 Significance

The high-Z cluster represents the **skill ceiling of the game** as observed in the corpus. It demonstrates that at the absolute top of the ladder, games converge to a specific structural archetype (long, tight, single-crown) regardless of the deck matchup. This has implications for model generalization: patterns learned from the main body (where decisive wins and blowout losses are common) may not transfer to this regime.

---

## 3. Tilt Cluster Identification

### 3.1 Cluster Profiling Method

After HDBSCAN clustering, each cluster is profiled by its win rate, average elixir leaked, and behavioral characteristics. Clusters are sorted by win rate to identify the worst-performing groups.

### 3.2 Core Tilt Clusters

| Cluster | Size | Win Rate | Avg Leak | Signature |
|---------|------|----------|----------|-----------|
| C5 | ~30 | 0% | 15.2 | Extreme leak, 3-crown losses |
| C10 | ~25 | 0% | 18.7 | Highest leak in corpus |
| C11 | ~20 | 0% | 12.4 | Consecutive loss patterns |
| C12 | ~15 | 0% | 14.8 | Defensive collapse |
| C13 | ~20 | 0% | 13.1 | Late-game meltdown |
| C14 | ~25 | 0% | 16.5 | Full tilt spiral |

**Common features:** 0% win rate, 12-20 elixir leaked (vs corpus average of ~3.0), overwhelmed-leg spatial pattern, low alternation rate (not reactive — frozen or panic-playing).

### 3.3 Extended Tilt Clusters

| Cluster | Size | Win Rate | Avg Leak | Signature |
|---------|------|----------|----------|-----------|
| C7 | ~40 | 5% | 5.8 | Early leak, partial recovery |
| C25 | ~30 | 4% | 6.2 | Bad matchup + leak |
| C28 | ~25 | 6% | 5.1 | Competitive start, late collapse |
| C32 | ~20 | 3% | 7.4 | Sustained elevated leak |
| C35 | ~15 | 5% | 5.5 | Narrow loss with waste |
| C36 | ~20 | 4% | 6.8 | Overtime leak |

**Common features:** Near-zero win rate (3-6%), 5-7 elixir leaked, mixed between contested and overwhelmed legs. These represent the *onset* of tilt — the player is struggling but not yet in full collapse.

### 3.4 Tilt as a Manifold Phenomenon

The tilt clusters are not randomly scattered across the embedding space. They occupy a specific region — predominantly the overwhelmed leg, with a gradient from extended tilt clusters (at the leg's base, near the contested region) to core tilt clusters (at the leg's extremity). This spatial coherence in 128-dim space is what enables embedding-based tilt detection: new games that land near this region's centroids are likely tilt games.

---

## 4. Player Comparison via Replay Telemetry

### 4.1 Method

Player profiles are constructed by aggregating replay event features across all of a player's games in the corpus. The profile includes play count, card type ratios, elixir discipline, aggression, tempo (inter-play gaps), and phase distribution.

### 4.2 Comparison: 람세스 vs KrylarPrime

람세스 (Ramses) climbed from 11,121 to 12,154 trophies — one of 5 opponents in the personal corpus who broke through to 12K+. Replay telemetry comparison:

| Metric | 람세스 | KrylarPrime |
|--------|--------|-------------|
| Deck archetype | MK/Wall Breakers/Bait | PEKKA/GY/Miner |
| Avg plays/game | 31.1 | 22.5 |
| Card type ratio | 93% troop | 73% troop |
| Elixir leaked/game | 5.1 | 2.0 |
| Aggression index | 42% | 53.5% |
| Median inter-play gap | 101 ticks | 130 ticks |
| First-mover rate | 68% | 54% |
| Win rate (in corpus) | 67% (18 games) | 62.5% (personal) |

**Interpretation:** 람세스 plays a fundamentally different game — higher tempo (31 vs 22 plays), faster cycling (101 vs 130 tick gaps), more reactive (93% troop deployments). The elixir leak is 2.5x higher, suggesting a "volume over precision" approach where some waste is acceptable because the tempo compensates. The aggression is actually *lower* than KrylarPrime's, which seems counterintuitive for a spam-heavy deck, but makes sense: Wall Breakers bait forces defensive card placements to control the opponent's pushes.

### 4.3 Comparison: logoman vs KrylarPrime

logoman (11,111 → 11,685) plays LavaLoon and has a remarkably similar telemetry profile to KrylarPrime:

| Metric | logoman | KrylarPrime |
|--------|---------|-------------|
| Avg plays/game | 24.6 | 22.5 |
| Elixir leaked/game | 1.8 | 2.0 |
| Aggression index | 57.2% | 53.5% |
| Median inter-play gap | 124 ticks | 130 ticks |
| Win rate | 59% | 62.5% |

**Interpretation:** logoman is a near-doppelganger in terms of tempo, discipline, and aggression. The LavaLoon archetype naturally maps to a similar playstyle — patient elixir management, decisive pushes, moderate tempo. The matchup data shows KrylarPrime now dominates LavaLoon (5-0 all-time against Lava Hound), suggesting that while the playstyles are similar, the specific deck interaction has evolved in KrylarPrime's favor.

---

## 5. Archetype Matchup Distribution

### 5.1 Method

Opponent decks are classified by win condition using a static archetype map (`archetypes.py`). Win rates are computed per archetype with Beta(1,1) prior for uncertainty quantification.

### 5.2 Challenging Archetypes (from personal data)

| Archetype | Record | Win Rate | Significance |
|-----------|--------|----------|-------------|
| Hog Cycle | 10-31 | 24% | Structural weakness — fast cycle outpaces the deck |
| Goblin Barrel Bait | 6-13 | 32% | Spell bait overwhelms available counters |
| Mega Knight | 11-19 | 37% | MK's area denial disrupts placement-dependent combos |

### 5.3 Dominant Archetypes

| Archetype | Record | Win Rate | Significance |
|-----------|--------|----------|-------------|
| Lava Hound | 5-0 | 100% | Complete air dominance |
| Golem Beatdown | majority wins | ~65% | Patient play punishes slow buildup |
| Bridge Spam | majority wins | ~60% | Reactive defense → counterattack |

### 5.4 Tilt Session Anatomy (Case Study)

A tilt session was identified in the last 10 games: 7 losses, 140-trophy drop (11,140 → 11,000).

**Sequence:**
1. takea18: 0-3 blowout (trigger game)
2-6. Six consecutive losses, five with 0-1 scores
7-10. Mixed (3 wins, 1 loss)

**Archetype distribution during tilt:**
- 2x Mortar Siege
- 1x Goblin Barrel Bait
- 1x Hog Cycle
- 2x other

The tilt was triggered by a blowout loss (0-3) followed by consecutive encounters with the player's worst archetypes. The 0-1 score pattern across five consecutive losses indicates the *games were close* — the deck was competitive, but the marginal execution quality required to convert close games was degraded by tilt state.

This pattern is consistent with the overwhelmed-leg profile: narrow losses with elevated (but not extreme) elixir leak, indicating that the fundamental strategy is sound but the player is making small execution errors that compound into losses.

---

## 6. Corpus Composition

### 6.1 Data Sources

| Source | Games | With Replays | Trophy Range | Purpose |
|--------|-------|-------------|--------------|---------|
| Personal (KrylarPrime) | ~2,800 | ~400 | 10,000-11,100 | Ground truth for personal patterns |
| Corpus (top-200 global) | ~8,000+ | ~3,400 | 7,000-13,000+ | Meta-level patterns, pre-training |

### 6.2 Replay Coverage

The bottleneck for ML model quality is replay coverage — only games with scraped replay events can be used for TCN training. The combined corpus scraper (chaining battle log fetch + replay scrape per player) maximizes the data window overlap, achieving 100% match rate on some players vs. the previous ~5% match rate when scraping battles and replays on independent schedules.

### 6.3 Trophy Distribution

The corpus is skewed toward the very top of the ladder:
- 70% of corpus games involve players above 8,000 trophies
- 30% above 11,000
- The high-Z outlier cluster is exclusively 12,000+

This means the TCN learns primarily from elite play patterns. Transfer to the personal data (~11,000 range) is reasonable given that both populations are in the top-ladder regime, but patterns specific to lower trophy ranges (where card levels and strategy quality differ significantly) are not represented.
