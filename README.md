# Clash Stats

Battle history tracker, analytics engine, and ML research platform for Clash Royale.

Built for players who care about data, not just meta. The CR API only exposes the last 25 battles with no historical queries. Community tools only capture games if they happen to poll during that window. This fills the gap: poll continuously, deduplicate, archive everything, scrape full replay telemetry, and surface analytics that RoyaleAPI/StatsRoyale/Deckshop don't provide.

## What It Does

**Personal battle database** — Fetches your battles every minute, deduplicates via SHA-256 hashing, and stores the full API response including tower HP, elixir leaked, and card levels. Never lose a game to the 25-battle API window again.

**Top-ladder training corpus** — Scrapes battle logs from 473+ tracked players (global top 200, regional leaderboards, opponent networks, and priority targets) every 30 minutes. Currently 17,000+ battles and growing at ~4,000/day.

**Replay event telemetry** — Uses Playwright and headless Chromium to scrape card placement events from RoyaleAPI replays. Every card play is captured with tick-level timing, arena coordinates, and elixir economy summaries. 22% replay coverage across the corpus and climbing.

**Monte Carlo matchup analysis** — Bayesian Beta-binomial posteriors for every archetype matchup, unsupervised sub-archetype detection via Jaccard-similarity clustering, and a full card interaction matrix showing P(win | opponent has card X).

**UMAP game embeddings** — 2D projections of game states with interactive click-to-explore similarity search. Gaussian kernel similarity with median-heuristic bandwidth, split views for corpus vs personal games.

**Live dashboard** — Flask app on port 8078 with trophy history, matchup breakdowns, recent battles, streak analysis, time-of-day performance, rolling window stats, and simulation results.

**Observability** — Prometheus metrics, Grafana dashboards, Loki log aggregation via Alloy, and Discord webhook alerting. Full pipeline health monitoring.

## Quick Start

### Prerequisites

- Python 3.11+
- API key from [developer.clashroyale.com](https://developer.clashroyale.com)
- Docker (for full stack with replay scraping and monitoring)

### Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Environment Variables

```bash
cp .env.example .env
# Edit .env with your credentials
```

| Variable | Required | Description |
|---|---|---|
| `CR_API_KEY` | Yes | API key from developer.clashroyale.com |
| `CR_PLAYER_TAG` | Yes | Player tag without `#` |
| `CR_API_URL` | No | API base URL (default: official API, or `https://proxy.royaleapi.dev/v1` to avoid IP whitelisting) |
| `GRAFANA_PASSWORD` | No | Grafana admin password (default: `changeme`) |
| `DISCORD_WEBHOOK_URL` | No | Discord webhook for pipeline alerts |

### Docker

Full stack: tracker, headless browser, Prometheus, Grafana, Loki, and Alloy.

```bash
docker compose up -d
```

| Service | Port | Purpose |
|---|---|---|
| `cr-tracker` | 8078 | Dashboard + API + metrics |
| `cr-browser` | 6080 | Headless Chromium (noVNC for replay auth) |
| `prometheus` | 9090 | Metrics storage |
| `grafana` | 3000 | Dashboards and alerting |
| `loki` | 3100 | Log aggregation |
| `alloy` | — | Docker log collector → Loki |

## CLI Reference

The `clash-stats` CLI is the primary interface for all operations.

### Personal Tracking

```bash
clash-stats --fetch                      # Fetch and store your latest battles
clash-stats --stats                      # Overall win rate, trophy range, totals
clash-stats --deck-stats                 # Per-deck breakdown
clash-stats --matchups                   # Card matchup analysis
clash-stats --crowns                     # Crown distribution
clash-stats --recent 20                  # Last N battles
clash-stats --streaks                    # Win/loss streak analysis
clash-stats --rolling 35                 # Rolling window stats
clash-stats --trophy-history             # Trophy progression over time
clash-stats --archetypes                 # Opponent archetype analysis
clash-stats --export json --output f.json # Export data as CSV or JSON
```

### Replay Scraping

```bash
clash-stats --replay-login               # Start RoyaleAPI auth (complete via noVNC on :6080)
clash-stats --replay-check               # Verify auth status
clash-stats --fetch-replays              # Scrape replays for your recent battles
```

### Corpus Management

```bash
clash-stats --corpus-update              # Refresh top-200 global leaderboard player tags
clash-stats --corpus-scrape              # Scrape battle logs for corpus players
clash-stats --corpus-replays             # Scrape replays for corpus battles
clash-stats --corpus-stats               # Show corpus size, coverage, source breakdown
clash-stats --corpus-add-priority TAG    # Promote a player to priority scraping (12x/day)
clash-stats --corpus-discover            # Discover new players from opponent tags in existing battles
clash-stats --corpus-locations           # Discover players from regional leaderboards
clash-stats --corpus-nemeses             # Add opponents you've lost to into the corpus
```

#### Scraping Options

```bash
--corpus-limit N                         # Max players to process per run (default: 20)
--replays-per-player N                   # Max replays per player (default: 25)
--max-pages N                            # Pagination depth (default: 5)
--concurrency N                          # Parallel browser tabs (default: 1)
```

### Simulation & Analysis

```bash
clash-stats --sim-matchups               # Bayesian matchup posteriors with sub-archetype breakdown
clash-stats --sim-interactions           # Card interaction matrix — P(win | opponent has card)
clash-stats --sim-full                   # Run full simulation suite and cache for dashboard
```

## Architecture

### Data Pipeline

```
CR Official API                RoyaleAPI (web scraping)
     │                                │
     ▼                                ▼
  Battle logs ──────────────► Replay events
  (25 max, polled 1m)         (Playwright/Chromium)
     │                                │
     ▼                                ▼
┌─────────────────────────────────────────────┐
│              SQLite Database                │
│                                             │
│  battles ──── deck_cards                    │
│      │                                      │
│      ├─── replay_events (tick/x/y/card)     │
│      └─── replay_summaries (elixir stats)   │
│                                             │
│  player_snapshots (trophy history)          │
│  player_corpus (473+ tracked players)       │
└──────────────────┬──────────────────────────┘
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
     Analytics  Simulation  Dashboard
                   │
              ┌────┼────┐
              ▼    ▼    ▼
          Matchup Card  Sub-archetype
          Model   Matrix Detection
```

### Corpus Discovery

Player discovery is multi-layered:

- **Global top 200** — weekly refresh from the CR API leaderboard
- **Regional leaderboards** — weekly discovery from location-based rankings
- **Opponent network** — daily crawl of opponent tags from existing battles
- **Nemeses** — hourly addition of players you've lost to
- **Priority queue** — manually promoted players scraped 12x/day (every 15 min via 4 concurrent tabs)

### Cron Schedule (Automated)

| Interval | Job | Description |
|---|---|---|
| Every 1m | `fetch` | Personal battle polling |
| Every 5m | `publish_stats` | Push dashboard JSON to GitHub `stats` branch |
| Every 15m | `corpus_replays_priority` | Priority player replay scraping |
| Every 30m | `corpus_scrape` | Corpus battle log scraping |
| 2x/hour | `corpus_replays` | Full corpus replay scraping (5 tabs) |
| 2x/hour | `replay_wrapper` | Personal replay scraping |
| Hourly | `corpus_nemeses` | Add lost-to opponents |
| Daily | `corpus_discover` | Network expansion from opponent tags |
| Weekly | `corpus_update` | Refresh global top-200 player list |
| Weekly | `corpus_locations` | Regional leaderboard discovery |
| Every 6h | `sim_refresh` | Recompute simulation results |

### Stats Publishing

Dashboard API responses are serialized to JSON and force-pushed as orphan commits to the `stats` branch every 5 minutes. This enables external consumers to read near-realtime game data without API access.

### Card Variant Tracking

The December 2025 update introduced Hero cards — variants of existing cards with activatable abilities (Hero Giant's Heroic Hurl, Hero Wizard's Fiery Flight, etc.). The schema distinguishes three card variants via `card_variant` in `deck_cards`:

| Variant | `evolution_level` | `maxEvolutionLevel` | `card_variant` |
|---|---|---|---|
| Base | 0 | — | `base` |
| Evolution | 1 | 1 | `evo` |
| Hero | 2 | 2-3 | `hero` |

## Simulation Framework

### Bayesian Matchup Model

Every archetype matchup is modeled as a Beta-binomial conjugate with a uniform prior: Beta(W+1, L+1). Posterior means, 95% credible intervals, and CI width as a data-sufficiency indicator. CI width > 0.20 = insufficient data for confident conclusions.

### Sub-Archetype Detection

Unsupervised decomposition of archetypes using greedy agglomerative clustering on Jaccard similarity of support card sets (threshold: 0.55). Discovers community-recognized deck variants and identifies unnamed emerging archetypes. Example: "Hog Rider" decomposes into 7 distinct clusters; "Miner" into 9 — evidence that single-card archetype labels are taxonomically insufficient.

### Card Interaction Matrix

Per-card P(win | opponent has card X) with Beta posteriors. Provides finer-grained matchup information than archetype-level aggregation. Can serve as zero-shot priors for unseen deck matchups via logistic regression on card-level win probabilities.

## ML Roadmap

Seven ADRs define the planned ML pipeline, from statistical baselines to deep learning:

| ADR | Title | Status |
|---|---|---|
| [001](docs/adr/001-feature-engineering.md) | Feature Engineering from Replay Events | Implemented (tabular features) |
| [002](docs/adr/002-monte-carlo-simulation.md) | Monte Carlo Simulation Framework | Implemented |
| [003](docs/adr/003-game-state-embeddings.md) | Game State Embeddings (UMAP → TCN → Transformer) | Phase 0 (UMAP) live |
| [004](docs/adr/004-win-probability-estimator.md) | Real-Time Win Probability Estimator | Planned |
| [005](docs/adr/005-opponent-prediction.md) | Opponent Play Prediction Model | Planned |
| [006](docs/adr/006-counterfactual-simulator.md) | Counterfactual Deck Simulator (CVAE) | Planned |
| [007](docs/adr/007-training-data-pipeline.md) | Training Data Pipeline & Scale Strategy | Implemented |
| [008](docs/adr/008-observability.md) | Observability & Alerting | Implemented |

### Guiding Principles

1. **Statistical models first, deep learning second.** Monte Carlo runs on day one with 200 games. Neural models need scale.
2. **Everything is queryable.** Model outputs go back into SQLite. Predictions are data, not console output.
3. **Train on the meta, fine-tune on you.** General models from the top-ladder corpus, specialized via transfer learning on personal games.
4. **Offline-first.** All training and inference runs locally. No cloud dependencies, no API costs.

## Observability

Full monitoring stack via Docker Compose:

- **Prometheus** — scrapes `/metrics` from the tracker every 15s
- **Grafana** — pre-provisioned dashboards for scrape health, corpus growth, API errors, replay coverage
- **Loki + Alloy** — centralized log aggregation from all containers
- **Discord webhooks** — alerting for scrape failures, API 429s, data freshness

Metrics include: battles scraped/deduped, API response times, replay parse success/failure rates, corpus size, Chromium tab utilization, and data freshness gauges.

## Project Structure

```
├── src/tracker/
│   ├── cli.py                  # CLI entrypoint (clash-stats command)
│   ├── analytics.py            # Query layer: stats, matchups, streaks, rolling windows
│   ├── api.py                  # CR API client with retry/backoff
│   ├── archetypes.py           # Win condition → archetype classifier
│   ├── corpus.py               # Corpus player discovery and management
│   ├── corpus_scraper.py       # Battle + replay scraping for corpus players
│   ├── dashboard.py            # Flask app: UI + REST API + Prometheus /metrics
│   ├── database.py             # SQLAlchemy engine/session factory
│   ├── export.py               # CSV/JSON export
│   ├── metrics.py              # Prometheus metric definitions + batch accumulator
│   ├── models.py               # ORM models (Battle, DeckCard, ReplayEvent, PlayerCorpus, etc.)
│   ├── replays.py              # Playwright-based replay scraper (RoyaleAPI)
│   ├── reporting.py            # Formatted text reports for CLI output
│   ├── simulation/
│   │   ├── interaction_matrix.py  # Card-level P(win|card) with Beta posteriors
│   │   ├── matchup_model.py       # Archetype matchup posteriors + sub-archetype detection
│   │   └── runner.py              # Orchestrator: run all simulations, cache results
│   ├── static/                 # Dashboard JS + CSS
│   ├── templates/              # Dashboard HTML
│   ├── alembic/                # Database migrations (5 versions)
│   └── tests/                  # 148 tests
├── docs/
│   ├── adr/                    # Architecture Decision Records (001-008)
│   ├── observability.md        # Monitoring architecture reference
│   └── research-notes.md       # Research paper findings and methodology
├── docker/
│   ├── browser/                # Playwright/Chromium container (noVNC)
│   └── monitoring/             # Prometheus, Grafana, Loki, Alloy configs
├── docker-compose.yml          # Full stack: tracker, browser, monitoring
├── crontab                     # Automated scrape/publish schedule
├── entrypoint.sh               # Container init: SSH, cron scripts, dashboard launch
├── publish_stats.sh            # Git-plumbing stats branch publisher
└── pyproject.toml
```

## Tests

```bash
pytest
```

148 tests covering the database layer, API client, analytics queries, reporting, CLI, dashboard API, replay parsing, and model relationships.

## Research

This project doubles as a research platform. Early findings documented in [docs/research-notes.md](docs/research-notes.md):

- **Unsupervised sub-archetype detection** — 60 clusters across 14 win conditions from 5,500+ battles, validating community taxonomy while discovering unnamed variants
- **Card interaction matrix as matchup prior** — card-level Bayesian posteriors provide finer signal than archetype aggregation
- **Bayesian vs raw win rate** — credible interval width as a decision criterion for data sufficiency

All analysis uses classical statistics and set-theoretic clustering with no machine learning, establishing a reproducible baseline against which learned representations can be evaluated as data scales.
