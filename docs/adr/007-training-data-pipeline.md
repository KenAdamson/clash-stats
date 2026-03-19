# ADR-007: Training Data Pipeline & Scale Strategy

**Status:** Implemented
**Date:** 2026-02-22
**Depends on:** Replay scraper (`replays.py`), ADR-001 (Feature Engineering)

## Context

Every ADR in this stack has a scale table, and the numbers are clear: personal replay data alone won't reach the volumes needed for the deep learning models. At 1-2 games per day, accumulating 2,000 games with replay data takes 3-4 years. The models need thousands of games *now*.

The solution: train on the broader meta. RoyaleAPI exposes replay data for every player, not just KrylarPrime. The top 10,000 ladder players collectively generate tens of thousands of games per week. That's the training corpus. Fine-tune on personal games for player-specific patterns.

This is transfer learning applied to game analytics: learn the general dynamics of Clash Royale from the population, then specialize to one player's style.

## Decision

### 1. Data Sources

#### Source A: KrylarPrime Personal Games
- **Volume:** ~1-2 games/day, ~500-700/year
- **Replay coverage:** 100% (all recent games scraped automatically)
- **Value:** Ground truth for personal patterns, deck-specific interactions, player-specific tendencies
- **Collection:** Existing pipeline (`--fetch` + `--fetch-replays`)

#### Source B: Top Ladder Replays
- **Volume:** Top 10,000 players × ~5 games/day = ~50,000 games/day
- **Replay coverage:** Sampling strategy needed (can't scrape 50K/day)
- **Value:** Dense interaction matrices, archetype coverage, meta-level patterns
- **Collection:** New scraper targeting top-ladder player tags

#### Source C: Specific Matchup Replays
- **Volume:** Targeted — search for games involving specific deck matchups
- **Replay coverage:** As needed
- **Value:** Fills sparse regions of the interaction matrix (e.g., rare card combinations)
- **Collection:** RoyaleAPI deck search → player tags → replay scrape

### 2. Top Ladder Scraping Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│ CR API           │     │ RoyaleAPI        │     │ PostgreSQL        │
│ /v1/locations/   │────►│ /player/{tag}/   │────►│ battles (JSONB)   │
│   global/        │     │   battles        │     │ replay_events     │
│   rankings       │     │ /replay?...      │     │ replay_summaries  │
│                  │     │                  │     │ player_corpus     │
└─────────────────┘     └──────────────────┘     └───────────────────┘
     Step 1:                 Step 2:                  Step 3:
  Fetch top player       Scrape battles &          Store with
     tags                  replay HTML              provenance
```

**Step 1: Tag collection**
- CR API endpoint: `GET /locations/global/rankings/players?limit=200`
- Returns top 200 players globally
- Also available per-region: US, EU, Asia
- Poll weekly — top ladder is relatively stable
- Store in a `player_corpus` table with `source = "top_ladder"` provenance

**Step 2: Battle & replay scraping**
- For each corpus player tag, scrape their recent battles via RoyaleAPI
- Rate limiting: 3-second delay between page loads (existing `FETCH_DELAY`)
- Session management: shared RoyaleAPI authentication via stored browser state
- Deduplication: same SHA-256 battle_id scheme as personal games

**Step 3: Storage with provenance**

```sql
ALTER TABLE battles ADD COLUMN corpus TEXT DEFAULT 'personal';
-- 'personal' = KrylarPrime's games
-- 'top_ladder' = scraped from top-200 players
-- 'matchup_targeted' = targeted scrape for specific matchups

CREATE TABLE player_corpus (
    player_tag TEXT PRIMARY KEY,
    player_name TEXT,
    source TEXT NOT NULL,          -- 'top_ladder', 'matchup_search', 'manual'
    trophy_range_low INTEGER,
    trophy_range_high INTEGER,
    games_scraped INTEGER DEFAULT 0,
    replays_scraped INTEGER DEFAULT 0,
    last_scraped DATETIME,
    active INTEGER DEFAULT 1       -- 0 = stop scraping this player
);
```

### 3. Sampling Strategy

Scraping 50K games/day is unnecessary and would abuse RoyaleAPI. Instead, use stratified sampling:

**Tier 1: Weekly full scrape (200 players)**
- Top 200 global leaderboard
- All recent battles + replays
- ~200 players × 25 battles/window × 1 scrape/week = 5,000 games/week
- Purpose: meta-level pattern learning, dense interaction matrices

**Tier 2: Archetype-targeted scraping (as needed)**
- Identify archetypes that are sparse in the personal corpus (e.g., Goblin Barrel bait, Inferno Dragon cycle)
- Search RoyaleAPI for top players running those archetypes
- Scrape their games specifically
- Purpose: fill gaps in the interaction matrix, improve matchup models for problem matchups

**Tier 3: Mirror matchup collection (rare)**
- Search for games where the opponent runs a deck similar to KrylarPrime's
- These are extremely rare (zero community usage per RoyaleAPI) but invaluable for understanding the mirror
- Purpose: edge case handling

**Estimated corpus growth (original projection, pre-implementation):**
- Month 1: 5,000 games (Tier 1 weekly scrapes)
- Month 3: 15,000 games + 2,000 targeted
- Month 6: 30,000+ games
- Year 1: 60,000+ games with full replay event data

**Actual corpus growth (measured 2026-02-23):**
- Day 1: 5,699 top-ladder battles from 200 players — exceeded Month 1 projection in 24 hours
- Growth rate: ~3,000-4,000 battles/day at 2-hour scrape intervals
- Projected Week 1: ~25,000 battles (5x the original Month 1 estimate)
- Projected Month 1: ~100,000+ battles

The original projections assumed weekly scraping of 200 players (1 scrape/week × 25 battles/window = 5,000/week). Actual deployment scrapes every 2 hours, capturing most of each player's daily output rather than a single 25-game window per week. The API's 25-battle buffer is the constraint — frequent scraping keeps the buffer from overflowing rather than increasing per-scrape yield.

**Scrape frequency evolution:**
- v1 (original ADR): weekly scrape → 5K/month
- v2 (initial deployment): every 6h → 5.7K/day
- v3 (current, 2026-02-23): every 2h → estimated 3-4K/day sustained

### 4. Data Quality and Filtering

Not all scraped games are useful for training. Filter criteria:

| Filter | Rationale |
|--------|-----------|
| `battle_type == "PvP"` only | Exclude 2v2, challenges, party modes — different game mechanics |
| Trophy range ≥ 7,000 | Below 7K, play patterns are too different from 11K+ to transfer |
| `replay_events.count >= 10` | Games with very few events are likely disconnects or instant surrenders |
| No duplicate deck matchups beyond 50/matchup | Prevent the model from overfitting to a single popular matchup |
| Balance win/loss ratio per archetype | Prevent class imbalance per matchup type |

### 5. Transfer Learning Pipeline

```
Phase 1: Pre-train on top-ladder corpus
  ├── Card embeddings (ADR-001)
  ├── TCN encoder (ADR-003)
  ├── Win probability model (ADR-004)
  └── Opponent prediction model (ADR-005)

Phase 2: Fine-tune on personal games
  ├── Freeze lower TCN layers, fine-tune upper layers + heads
  ├── Player-specific card interaction patterns
  ├── Player-specific positioning tendencies
  └── Personal deck's unique matchup dynamics

Phase 3: Continuous learning
  ├── New personal games → incremental fine-tuning
  ├── New top-ladder games → periodic pre-training refresh
  └── Meta shift detection → trigger full retrain
```

**Fine-tuning strategy:**
- Freeze the card embedding layer and first 3 TCN blocks (general game dynamics)
- Fine-tune the last 3 TCN blocks and all prediction heads (player-specific patterns)
- Learning rate: 10x lower than pre-training (prevent catastrophic forgetting)
- Early stopping on personal validation set (last 20% of personal games chronologically)

### 6. Meta Shift Detection

The Clash Royale meta shifts with balance patches (typically every 1-3 months) and seasonal changes. A model trained on pre-patch data may be miscalibrated post-patch.

**Detection method:**
- Track the model's prediction accuracy on new games in a rolling 50-game window
- If accuracy drops below a threshold (e.g., >10% below the validation baseline for 20+ games), flag a potential meta shift
- Compare card frequency distributions in recent top-ladder scrapes vs the training set — significant KL divergence in card usage indicates a meta shift

**Response to detected shift:**
1. Scrape new top-ladder data post-patch
2. Retrain from scratch (or from a checkpoint prior to fine-tuning)
3. Re-fine-tune on personal games
4. Re-compute all embeddings and predictions

### 7. Ethical and Practical Considerations

**Rate limiting:** RoyaleAPI is a community resource. The scraper must be respectful:
- 3-second minimum delay between page loads (already implemented)
- 5-second delay between switching players
- Per-run safety cap (5,000 replays) — real throttle is Cloudflare/RoyaleAPI pushback
- Backoff on HTTP 429 (rate limit) responses
- Identify the scraper via User-Agent string

**Data usage:** All scraped data is used for personal analytics only. No redistribution, no public dashboards showing other players' data. The corpus is a training set, not a surveillance tool.

**Storage:** At 60K games/year with ~50 events each, the replay_events table grows by ~3M rows/year. PostgreSQL with TOAST handles this well — `raw_json` JSONB values are stored out-of-line, keeping table scans fast. The feature cache (.npz files) may reach 5-10 GB. Archive older feature caches when models are retrained.

**ITAR note:** None of this involves controlled technical data. Game analytics is firmly in the public domain. But the engineering discipline (comprehensive data collection, provenance tracking, version-controlled models) reflects the same rigor.

### 8. Implementation

```
src/tracker/
├── corpus.py              ← Top-ladder tag collection, corpus management
├── corpus_scraper.py      ← Batch replay scraping for corpus players
└── meta_shift.py          ← Meta shift detection, retrain triggering
```

New tables in migration 004:
```sql
-- player_corpus table (as defined in §2)
-- ALTER battles ADD corpus column
```

CLI additions:
```
clash-stats --corpus-update              # Refresh top-ladder player tags
clash-stats --corpus-scrape [--limit N]  # Scrape replays for corpus players
clash-stats --corpus-stats               # Show corpus composition and coverage
clash-stats --meta-check                 # Run meta shift detection
```

Crontab additions:
```
# Weekly: refresh top-ladder tags
0 6 * * 1  clash-stats --corpus-update

# Daily: scrape top-ladder replays (limit 200/day)
0 8 * * *  clash-stats --corpus-scrape --limit 200

# Weekly: check for meta shifts
0 12 * * 5  clash-stats --meta-check
```

## Consequences

### Positive
- Solves the cold-start problem for all neural models — 5,000 games in month 1 instead of waiting years
- Transfer learning from meta to personal is a well-understood technique with strong theoretical backing
- Archetype-targeted scraping fills exactly the gaps that matter most (problem matchups)
- Meta shift detection prevents model staleness after balance patches
- Corpus provenance tracking enables per-source analysis: "how well does top-ladder data predict 11K outcomes?"

### Negative
- Dependency on RoyaleAPI's HTML structure for the broader corpus (same risk as personal scraping, but amplified)
- RoyaleAPI rate limits may constrain scraping volume — 1,000 replays/day may not be achievable
- Top-ladder play patterns (8,000+ trophies) may not transfer perfectly to 11K play — skill level and card levels differ
- Storage and compute requirements scale linearly with corpus size
- Risk of training on stale post-patch data if meta shift detection has lag

### Projected Corpus Timeline

**Original projections (pre-implementation):**

| Milestone | Personal Games | Top-Ladder Games | Total | Models Unlocked |
|-----------|---------------|-----------------|-------|----------------|
| Week 1 | 200 | 1,000 | 1,200 | Monte Carlo (ADR-002), Basic embeddings (ADR-003) |
| Month 1 | 250 | 5,000 | 5,250 | Win probability (ADR-004), Opponent prediction (ADR-005) |
| Month 3 | 350 | 17,000 | 17,350 | Counterfactual generation (ADR-006) |
| Month 6 | 500 | 35,000 | 35,500 | Full manifold exploration, deck gradient |
| Year 1 | 700 | 65,000 | 65,700 | Per-card interaction embeddings, meta evolution tracking |

**Revised projections (based on actual Day 1 data, 2026-02-23):**

| Milestone | Personal Games | Top-Ladder Games | Total | Models Unlocked |
|-----------|---------------|-----------------|-------|----------------|
| Day 1 ✓ | 140 | 5,699 | 5,839 | Monte Carlo (ADR-002) — **implemented** |
| Week 1 | 150 | 25,000 | 25,150 | Basic embeddings, UMAP manifold exploration |
| Month 1 | 200 | 100,000 | 100,200 | All sequence models, dense interaction matrices |
| Month 3 | 300 | 300,000 | 300,300 | Per-card embeddings, meta evolution tracking |

All neural model scale thresholds (ADR-003 through ADR-006) are expected to be met within 2 weeks rather than 3-6 months. The bottleneck has shifted from data volume to replay coverage and engineering time to build the models. Note: the RoyaleAPI proxy (`proxy.royaleapi.dev`) is currently Cloudflare-blocked (Error 1010); replay scraping uses HTTP-based fetching directly from RoyaleAPI's web interface.
