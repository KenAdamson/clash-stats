# Similarity Search & Tilt Detection

**Version:** 1.0
**Date:** 2026-03-01

## 1. Similarity Search

**Implementation:** `src/tracker/ml/similarity.py`

### 1.1 Problem Statement

Given a reference game, find the $k$ most similar games in the corpus. "Similar" is defined operationally: games that were *played* similarly (same tempo, aggression pattern, elixir economy, card usage distribution) regardless of the specific cards or outcome.

This answers the question: "I just played a weird game — have I seen this pattern before? What happened in those games?"

### 1.2 Distance Metric: Standardized Euclidean

The similarity search operates on the 50-dimensional tabular feature vectors (Phase 0), not on the learned TCN embeddings. This is a deliberate choice:

1. **Interpretability:** Each dimension of the 50-dim vector has a known meaning (card play count, aggression index, elixir leaked, etc.). Similar games found via these features are similar in *specific, nameable ways*.
2. **Availability:** Feature vectors exist for all games with replay data, regardless of whether the TCN has been trained.
3. **Stability:** Feature vectors don't change when the model is retrained. Similarity results are reproducible.

**Standardization:**

$$d(\mathbf{x}, \mathbf{y}) = \left\| \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}} - \frac{\mathbf{y} - \boldsymbol{\mu}}{\boldsymbol{\sigma}} \right\|_2$$

The `StandardScaler` is fit on the entire feature matrix (all games), then both the reference and candidate vectors are transformed. This ensures all 50 dimensions contribute proportionally:

```python
scaler = StandardScaler()
scaled = scaler.fit_transform(matrix)      # (n_games, 50)
ref_scaled = scaled[ref_idx]                # (50,)
distances = np.linalg.norm(scaled - ref_scaled, axis=1)  # (n_games,)
```

### 1.3 Similarity Metrics

Two complementary metrics are reported for each result:

#### Percentile Rank

The percentile rank answers: "What fraction of all games are *further away* than this one?"

$$\text{percentile}(i) = 1 - \frac{\text{rank}(i)}{n - 1}$$

where $\text{rank}(i)$ is the position in the distance-sorted order (0 = closest). A percentile of 0.99 means "Top 1% — closer than 99% of all games."

This metric is immediately human-readable and robust to changes in the feature distribution.

#### Gaussian Kernel Similarity

The Gaussian (RBF) kernel provides a natural [0, 1] similarity score adapted to the data distribution:

$$K(d) = \exp\left(-\frac{d^2}{2\sigma^2}\right)$$

where $\sigma$ is the **median distance** across all non-zero pairwise distances — an adaptive bandwidth parameter. Using the median rather than a fixed bandwidth ensures the kernel scale adjusts to the data:

```python
sigma = float(np.median(distances[distances > 0]))
gaussian = np.exp(-distances**2 / (2 * sigma**2))
```

- $K = 1.0$: identical game (distance = 0)
- $K \approx 0.61$: distance equals the median ($d = \sigma$)
- $K \approx 0.14$: distance equals $2\sigma$
- $K \to 0$: very dissimilar

The median-based bandwidth means that a kernel value of 0.8+ represents a genuinely close match in the context of the full corpus.

### 1.4 Result Structure

Results are split into two categories:
- **Personal games:** Similar games from the player's own battle history
- **Corpus games:** Similar games from the top-ladder corpus

This split enables two different analyses:
- Personal matches answer: "When have *I* played a game like this before, and what happened?"
- Corpus matches answer: "What do top-ladder players' games look like when they play this way?"

Each result includes:
- `battle_id`, `percentile`, `similarity` (kernel), `cluster_id`
- Battle metadata: `result`, `player_crowns`, `opponent_crowns`, `opponent_name`, `trophies`, `battle_time`
- Opponent deck: card names, classified archetype

### 1.5 Enrichment: `_enrich_results()`

The similarity search returns bare battle_ids with distance metrics. The enrichment step joins against the `battles` and `deck_cards` tables to add:

- Battle outcome and crown score
- Opponent name and trophy level
- Opponent deck composition (8 cards)
- Archetype classification (from `archetypes.py`)
- Corpus provenance (`personal` vs `corpus`)

This transforms raw similarity results into actionable game analysis.

---

## 2. Tilt Detection

**Implementation:** `src/tracker/ml/tilt_detector.py`

### 2.1 Problem Statement

Tilt is the competitive gaming equivalent of emotional decision-making: a player who is losing begins playing worse *because* they are losing, creating a negative feedback loop. In Clash Royale, tilt manifests as:

- **Elixir leaking:** Sitting at 10/10 elixir without playing, wasting generation
- **Consecutive losses:** Each loss compounds the emotional state
- **Score pattern changes:** Shifting from decisive wins (3-0, 3-1) to narrow losses (0-1)
- **Aggression collapse:** Retreating to defensive play when the deck is designed for offense

The tilt detector identifies these patterns from the most recent games and escalates warnings.

### 2.2 Two-Layer Detection Architecture

#### Layer 1: Heuristic Detection

Works immediately from battle metadata — no replay data or embeddings required.

**Signals monitored:**

| Signal | Threshold | Severity |
|--------|-----------|----------|
| Consecutive losses | ≥ 5 | Severe |
| Consecutive losses | ≥ 3, with avg leak ≥ 12.0 | Severe |
| Consecutive losses | ≥ 3 | Tilting |
| Tilt-pattern games in lookback | ≥ 4/10 | Tilting |
| Consecutive losses ≥ 2, avg leak ≥ 6.0 | — | Warning |
| Tilt-pattern games in lookback | ≥ 3/10 | Warning |
| Embedding tilt matches | ≥ 3/10 | Warning |

A **tilt-pattern game** is defined as a loss where either:
- Elixir leaked ≥ `LEAK_ELEVATED` (6.0), or
- Opponent achieved 3 crowns (complete destruction)

**Threshold calibration:** The leak thresholds are derived from TCN cluster analysis:
- `LEAK_SEVERE = 12.0`: Core tilt clusters (C5, C10, C11, C12, C13, C14) average 12-20 leaked elixir with 0% win rate
- `LEAK_ELEVATED = 6.0`: Extended tilt clusters (C7, C25, C28, C32, C35, C36) average 5-7 leaked elixir with ~5% win rate

**Lookback window:** The last 10 PvP ladder games (configurable via `LOOKBACK_GAMES`).

#### Layer 2: Embedding-Based Detection

When TCN embeddings exist, the detector counts how many recent games have cluster assignments matching known tilt clusters:

```python
CORE_TILT_CLUSTERS = {5, 10, 11, 12, 13, 14}       # 0% WR, 12+ leaked
EXTENDED_TILT_CLUSTERS = {7, 25, 28, 32, 35, 36}    # ~5% WR, 5+ leaked

all_tilt = CORE_TILT_CLUSTERS | EXTENDED_TILT_CLUSTERS
matches = sum(1 for _, cid in rows if cid in all_tilt)
```

These cluster IDs are identified from the TCN Phase 1 cluster profiling — they are the clusters where games exhibit the behavioral signature of tilt (extreme elixir leaking, near-zero win rate, defensive collapse).

**Why two layers:** The heuristic layer provides immediate detection from battle metadata alone — it works even without replay data or a trained model. The embedding layer adds confidence when available: a game that both has high elixir leak *and* falls near a tilt cluster centroid in 128-dim space is almost certainly a tilt game, not just a bad matchup.

### 2.3 Severity Levels

| Level | Meaning | Recommended Action |
|-------|---------|-------------------|
| `none` | No tilt detected | Continue playing |
| `warning` | Early signs — emerging pattern | Monitor closely |
| `tilting` | Active tilt — performance degrading | Take a break |
| `severe` | Full tilt — hemorrhaging trophies | Stop playing immediately |

### 2.4 TiltStatus Data Structure

```python
@dataclass
class TiltStatus:
    level: str                    # "none", "warning", "tilting", "severe"
    consecutive_losses: int       # Current loss streak
    recent_record: str            # "2W-5L"
    avg_leak_recent: float        # Mean elixir leaked over lookback window
    max_leak_recent: float        # Peak leak in any single game
    tilt_game_count: int          # Games matching tilt pattern in lookback
    embedding_matches: int        # Games near tilt centroids (0 if no embeddings)
    message: str                  # Human-readable status message
```

### 2.5 Decision Logic

The detection logic uses a priority cascade — the first matching condition determines the level:

```
1. consecutive_losses ≥ 5                              → SEVERE
2. consecutive_losses ≥ 3 AND avg_leak ≥ 12.0          → SEVERE
3. consecutive_losses ≥ 3                              → TILTING
4. tilt_pattern_games ≥ 4/10                           → TILTING
5. consecutive_losses ≥ 2 AND avg_leak ≥ 6.0           → WARNING
6. tilt_pattern_games ≥ 3/10                           → WARNING
7. embedding_matches ≥ 3/10                            → WARNING
8. None of the above                                   → NONE
```

### 2.6 Terminal Output

The `print_tilt_warning()` function renders the status with severity-appropriate formatting:

```
🔴 TILTING
  3 consecutive losses (2W-5L in last 7). Take a break.
  Recent: 2W-5L | Streak: 3L | Leak: 6.2 avg / 12.1 max
  TCN tilt cluster matches: 2
```

### 2.7 Dashboard Integration

The tilt detector exposes status via the `/api/tilt` endpoint, which the dashboard renders as a color-coded panel:
- None: hidden
- Warning: yellow indicator
- Tilting: red indicator
- Severe: red with pulsing animation

---

## 3. Mathematical Appendix

### 3.1 Gaussian Kernel as Similarity Measure

The Gaussian kernel $K(d) = \exp(-d^2 / 2\sigma^2)$ has several desirable properties as a similarity measure:

1. **Bounded:** $K \in (0, 1]$ with $K = 1$ iff $d = 0$.
2. **Monotone decreasing:** $\frac{dK}{dd} < 0$ for $d > 0$.
3. **Positive definite:** The Gaussian kernel is a valid Mercer kernel, meaning it corresponds to an inner product in a (infinite-dimensional) RKHS.
4. **Smooth:** Infinitely differentiable everywhere.
5. **Scale-adaptive:** The median-based $\sigma$ ensures the kernel is calibrated to the data distribution, not an arbitrary scale.

### 3.2 Percentile Rank vs. Kernel Similarity

The two metrics capture different information:

- **Percentile rank** is a *relative* measure — "this game is closer than X% of all games." It's robust to changes in the feature distribution but provides no absolute scale.
- **Kernel similarity** is an *absolute* measure — "the features of this game deviate from the reference by D standard deviations." It provides a consistent scale but is sensitive to the feature distribution.

In practice, both are reported because they answer different questions:
- "Is this game unusually similar?" → percentile rank
- "How similar, exactly?" → kernel similarity

### 3.3 Tilt as a Dynamical System Phenomenon

Tilt can be modeled as a positive feedback loop in a dynamical system:

1. **State:** Player emotional/cognitive state $s_t \in [0, 1]$, where 0 = optimal play, 1 = full tilt.
2. **Input:** Game outcome $o_t \in \{0, 1\}$ (loss/win).
3. **Dynamics:** $s_{t+1} = \alpha \cdot s_t + \beta \cdot (1 - o_t) \cdot f(s_t)$, where $f(s_t)$ is an increasing function (tilt compounds) and $\alpha < 1$ is a decay factor (tilt recovers over time/rest).
4. **Output:** Performance degradation manifests as elixir leak $\ell_t \propto s_t$ and win probability $P(\text{win}) \propto 1 - s_t^2$.

The tilt detector's heuristic signals (consecutive losses, leak accumulation) are observable proxies for the latent state $s_t$. The embedding-based detection provides a higher-dimensional observation by checking whether the game's behavioral profile matches the tilt cluster region of the embedding manifold — effectively a learned detector for the output of this dynamical system.
