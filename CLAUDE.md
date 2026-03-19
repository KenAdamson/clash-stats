# CLAUDE.md — Clash Royale Analytics Suite

## Project Overview

A Python tool for analyzing Clash Royale gameplay at a level the community ecosystem doesn't support. This isn't a hobbyist script — it's the analytics backend for a player who has climbed to 11,400+ trophies (PB 11,461 — 39 from ranked) with a deck that literally zero other humans play, in ~2,800 lifetime games while peers at the same trophy range have 10,000-15,000+.

The existing community tools (RoyaleAPI, StatsRoyale, Deckshop) are built for meta players running popular decks. They show aggregate stats, not individualized probabilistic analysis. This tool fills that gap.

### `cr_tracker.py` — Battle History Archiver

The CR API only exposes the last 25 battles. No pagination, no historical queries. Community sites only capture games if they happen to poll your tag during the window. This tool polls the API on a schedule, deduplicates via SHA-256 hashing, and archives every battle into PostgreSQL with full card-level data for long-term analysis.

**Intended deployment:** Docker container on a home server alongside an existing media stack. Third-party packages are welcome — use the best tool for the job (e.g., `requests` for HTTP, `rich` for terminal output, `pytest` for testing).

## Deck & Strategy

Deck composition and strategy details are intentionally excluded from this public repo. The deck has zero community usage on RoyaleAPI — that invisibility is a competitive advantage.

The tracker is designed to work with any 8-card deck. Deck data is pulled from the API at runtime.

### Evo Tracking

Only two Evo slots are available. The tracker's `_generate_deck_hash()` includes `evolutionLevel` in the hash, so different Evo configurations produce distinct deck hashes. The `deck_cards` table also stores `evolution_level` and `star_level` per card.

## Architecture Decisions

### Use the best packages available
Third-party dependencies are fine everywhere — tracker, analytics, dashboard. Use `requests` over `urllib`, `rich` for terminal output, `pytest` for testing, `pandas` for data analysis, etc. The Docker image handles `pip install`. Pick the best tool for the job.

### SQLAlchemy + Alembic
ORM models in `models.py`, versioned migrations via Alembic. The `database.py` module handles engine/session creation and runs migrations automatically on startup. For existing pre-Alembic databases, the migration runner stamps the initial revision without re-creating tables. Installed as a proper package with `pip install .` and a `clash-stats` CLI entrypoint.

### PostgreSQL as the data store
PostgreSQL 16 with JSONB for the `raw_json` columns. TOAST automatically stores large JSON values out-of-line, so full table scans on non-JSON columns are fast. The `raw_json` column preserves the complete API response as native JSONB — queryable with `->>`/`@>` operators and GIN-indexable. Connection pooling via SQLAlchemy (`pool_size=20, max_overflow=20`). Tests use SQLite in-memory via SQLAlchemy's `JSON` type fallback.

### Deduplication via content hashing
The API returns overlapping windows (last 25 battles). The SHA-256 hash on `battleTime + tags + crowns` handles dedup cleanly. Don't switch to timestamp-only dedup — the current approach is more robust.

## Implemented Features

All original known issues have been resolved:

1. **Deck hash includes Evo status.** `_generate_deck_hash()` now includes `evolutionLevel` — different Evo configurations produce distinct hashes. Existing data is backfilled on migration.
2. **Evo/star tracking in deck_cards.** `evolution_level` and `star_level` columns added, populated from API data. Schema migration backfills from `raw_json`.
3. **Trophy progression** — `--trophy-history` shows an ASCII chart of trophy movement over time.
4. **Streak detection** — `--streaks` shows current streak, longest win/loss runs with trophy ranges.
5. **Rolling window stats** — `--rolling N` shows win rate over the last N games with comparison to overall.
6. **Opponent archetype clustering** — `--archetypes` classifies opponent decks by win condition (Golem, Hog, etc.) and shows per-archetype win rates.
7. **Elixir leak tracking** — `player_elixir_leaked` and `opponent_elixir_leaked` stored per battle.
8. **Battle duration tracking** — `battle_duration` stored when available from the API.
9. **Snapshot diffing** — Each `--fetch` now prints changes since last fetch (trophy/win/loss deltas).
10. **Export capability** — `--export csv` or `--export json` with optional `--output FILE`. Works with any analytics command.

### Schema Migrations (Alembic)

Database migrations are managed by Alembic. Migrations were consolidated into a single `001_initial_schema.py` during the PostgreSQL migration. New migrations go in `src/tracker/alembic/versions/` starting from `002`. Run `alembic upgrade head` programmatically on startup via `init_db()`.

## Coding Standards

- **Python 3.11+.** Target the Docker image runtime.
- **Type hints everywhere.** Type hints aren't optional — they're documentation that the interpreter can validate.
- **Docstrings on all public methods.** Google style.
- **No silent failures.** If an API call fails or data is malformed, log it clearly. This runs unattended in Docker — failures must be visible in `docker logs`.
- **Conservative error handling.** Better to skip a malformed battle and log a warning than crash the whole fetch cycle.
- **Tests welcome.**

## Player Context for Smart Analytics

These data points inform what analytics matter:

- **Play style:** ~17 games/day in the current active stretch (434 games in 25 days). The ~1.35 games/day lifetime average is misleading — it includes an 8-9 year break from the game.
- **Efficiency:** ~2,800 games to 11,400+ trophies (PB 11,461). Peers at same range: 10,000-15,000+ games.
- **Three-crown rate:** ~73% lifetime (overwhelmingly wins by destruction, not chip)
- **Matchup data matters.** Both problem matchups and hard counters exist — the tracker should surface these from real battle data rather than relying on assumptions.
- **Tilt pattern:** Rare but devastating.

## API Reference

**Base URL:** `https://api.clashroyale.com/v1` (direct). The RoyaleAPI proxy (`proxy.royaleapi.dev`) is Cloudflare-blocked — use the official API only.

**Key endpoints:**
- `GET /players/{tag}` — Player profile (all-time stats, current trophies, clan)
- `GET /players/{tag}/battlelog` — Last 25 battles (the polling target)
- `GET /players/{tag}/upcomingchests` — Chest cycle (not currently used)

**Auth:** Bearer token in Authorization header. Key from developer.clashroyale.com.

**Rate limits:** Be respectful. The proxy has its own limits.

**Player tag encoding:** Tags start with `#` which must be URL-encoded as `%23`. The current code handles this.

## ML & Simulation Layer

The replay scraper (`replays.py`) transforms this from a results database into a full game telemetry system. Architecture Decision Records for the ML and simulation capabilities are in [`docs/adr/`](docs/adr/README.md):

- **Monte Carlo Simulation** (ADR-002): **Implemented.** Elixir economy modeling, opening hand analysis, Bayesian matchup estimation, card interaction matrices. CLI: `--sim-matchups`, `--sim-interactions`, `--sim-elixir`, `--sim-hands`, `--sim-full`.
- **Game State Embeddings** (ADR-003): **Phase 0+1 Implemented.** 50-dim feature extraction, two-stage UMAP (50→15→3), HDBSCAN clustering, TCN encoder (6-layer causal, 256-dim). Interactive 3D scatter plot on dashboard. Phase 2 (Transformer) pending 10K+ replay games.
- **Win Probability Estimator** (ADR-004): **Implemented (v2).** Causal TCN producing P(win) at every game tick. WPA per card placement. Platt-calibrated (ECE=0.031). 78.4% accuracy on 37.9K corpus games. Dashboard: interactive P(win) curves, card WPA tables with archetype drill-down. Incremental inference via 5-min cron. CLI: `--train-wp`, `--wp-infer-new`, `--wp BATTLE_ID`, `--wp-critical`, `--wp-cards`.
- **Opponent Prediction** (ADR-005): Proposed. Sequence model predicting opponent's next card, timing, and position.
- **Counterfactual Simulator** (ADR-006): Proposed. CVAE generating synthetic game sequences under deck modifications.
- **Training Data Pipeline** (ADR-007): **Implemented.** Top-ladder corpus (13K+ players), 3-4K battles/day, stratified sampling, transfer learning to personal games.
- **Observability** (ADR-008): **Implemented.** Prometheus metrics, Loki logs, Grafana dashboards, circuit breakers, structured retries.
- **Visual Game State Recognition** (ADR-009): **In Progress.** Replay-guided labeling (Phase 1.5), SAMv2 unit tracking on Arc A770 XPU (Phase 2). YOLO distillation pending.

Dependencies: `torch`, `numpy`, `scikit-learn`, `umap-learn`, `hdbscan`, `pandas` (ML extras installed via `pip install .[ml]` in the Docker image).

## Future Vision

The endgame is a unified analytics platform that runs as a Docker container with a lightweight web dashboard. Data-driven competitive advantage.

Priorities:
1. ~~Fix the Evo tracking gap in the tracker (deck hash + schema)~~ Done
2. ~~Add streak detection and rolling window stats~~ Done
3. ~~Build a simple web dashboard~~ Done (Flask + Chart.js + Plotly.js)
4. ~~Game state embeddings Phase 0+1 (ADR-003)~~ Done — feature extraction, UMAP, clustering, TCN encoder, 3D manifold visualization
5. ~~Win Probability Estimator (ADR-004)~~ Done (v2) — Platt-calibrated P(win) at every tick, 78.4% accuracy, dashboard visualization, incremental cron inference
6. ~~Monte Carlo simulation framework (ADR-002)~~ Done — elixir economy, opening hands, Bayesian matchups, interaction matrices
7. ~~Top-ladder corpus collection (ADR-007)~~ Done — 13K+ players, 3-4K battles/day
8. ~~Observability (ADR-008)~~ Done — Prometheus + Loki + Grafana + Alloy, circuit breakers, structured retries
9. Add tilt detection — if the tracker sees 3+ consecutive losses, surface a "you're tilting" warning
10. Complete BVT on replay scraper pipeline
11. Visual game state recognition (ADR-009) — SAMv2 tracking operational, YOLO distillation next
12. Game state embeddings Phase 2 (ADR-003) — Transformer encoder, pending 10K+ replay games

## Deployment

Docker Compose stack on a home server (local NVMe machine at 192.168.7.58).

### Services

| Service | Container | Purpose |
|---------|-----------|---------|
| `postgres` | `clash-postgres` | PostgreSQL 16 — primary data store |
| `cr-tracker` | `cr-tracker` | Battle tracker, dashboard, cron jobs |
| `cr-browser` | `cr-browser` | Headless Chromium sidecar for replay scraping |
| `cr-samv2` | `cr-samv2` | SAMv2 tracking API on Intel Arc A770 XPU |
| `prometheus` | `prometheus` | Metrics collection |
| `loki` | `loki` | Log aggregation |
| `alloy` | `alloy` | Docker log shipping to Loki |
| `grafana` | `grafana` | Dashboards and alerting |

### Docker Container Design

```
clash-stats/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── crontab                ← Debian cron schedule for all periodic jobs
├── entrypoint.sh          ← Container startup: fetch, dashboard, cron
├── publish_stats.sh       ← Git-push stats JSON to GitHub
├── docs/
│   └── adr/               ← Architecture Decision Records for ML/simulation layer
├── docker/
│   ├── browser/           ← Headless Chromium + NoVNC sidecar
│   ├── samv2/             ← SAMv2 tracking sidecar (Intel Arc XPU)
│   └── monitoring/        ← Prometheus, Loki, Alloy, Grafana configs
├── src/
│   └── tracker/
│       ├── __init__.py
│       ├── __main__.py          ← python -m tracker
│       ├── models.py            ← SQLAlchemy ORM (Battle, PlayerSnapshot, DeckCard, ReplayEvent, ReplaySummary)
│       ├── database.py          ← Engine/session setup, Alembic migration runner
│       ├── api.py               ← ClashRoyaleAPI client
│       ├── analytics.py         ← All query/storage functions
│       ├── replays.py           ← RoyaleAPI replay scraper (Playwright)
│       ├── replay_http.py       ← HTTP-based replay fetcher (replaces Playwright)
│       ├── corpus_scraper.py    ← Combined corpus pipeline: battles + replays
│       ├── reporting.py         ← Terminal output formatting
│       ├── dashboard.py         ← Flask web dashboard
│       ├── export.py            ← CSV/JSON export
│       ├── archetypes.py        ← Opponent deck classification
│       ├── metrics.py           ← Prometheus metrics + accumulated JSON flush
│       ├── cli.py               ← argparse + main() dispatch
│       ├── alembic/             ← Alembic migration config + versions (consolidated 001)
│       ├── simulation/          ← Monte Carlo framework (ADR-002)
│       ├── ml/                  ← ML: embeddings, win probability, clustering, activity
│       │   ├── card_metadata.py ← CardVocabulary — dynamic card→index mapping from DB
│       │   ├── features.py      ← 50-dim feature extraction from replay data
│       │   ├── sequence_dataset.py ← SequenceDataset — replay events → padded tensors
│       │   ├── umap_embeddings.py ← Two-stage UMAP (50→15→3) + supervised fitting
│       │   ├── clustering.py    ← HDBSCAN clustering + cluster profiling
│       │   ├── similarity.py    ← Euclidean distance + percentile rank + Gaussian kernel
│       │   ├── storage.py       ← GameFeature/GameEmbedding ORM models, numpy↔BLOB
│       │   ├── win_probability.py ← WinProbabilityModel — causal TCN architecture
│       │   ├── wp_training.py   ← WPTrainer, train/infer pipelines
│       │   ├── wp_dataset.py    ← Collate function for variable-length sequences
│       │   ├── wp_storage.py    ← WinProbability, GameWPSummary ORM models
│       │   ├── calibration.py   ← PlattCalibrator — Platt scaling + ECE diagnostics
│       │   └── activity_model.py ← GBM activity predictor for corpus scheduling
│       ├── vision/              ← Visual game state recognition (ADR-009)
│       └── tests/
├── data/                  ← Volume mount: ML models, metrics, session state
│   ├── postgres-local/    ← PostgreSQL data directory
│   └── ml_models/         ← Trained model checkpoints (.pt, .pkl)
└── .env                   ← CR_API_KEY, CR_PLAYER_TAG, DB_PASSWORD (not committed)
```

**Container requirements:**
- Base image: `python:3.11-slim-bookworm` with Intel Level Zero runtime for XPU
- Dependencies managed via `pyproject.toml`, installed with `pip install .[ml]` in the Dockerfile
- PostgreSQL data at `./data/postgres-local`, ML models at `./data/ml_models`
- Environment variables via `.env` file: `CR_API_KEY`, `CR_PLAYER_TAG`, `DB_PASSWORD`
- Debian cron schedule: personal combined every 2 min, corpus combined every 1 min (50 players/batch)
- Flask dashboard on port 8078, Prometheus metrics on port 8001
- Logging to stdout so Docker's log driver captures it

**Health/monitoring:**
- Log each fetch cycle with timestamp, new battle count, and current trophy count
- If the API returns errors for 3+ consecutive fetches, log a clear warning (don't silently fail — this runs unattended)
- Prometheus metrics: battle counts, replay counts, batch yields, rate limit backoffs, scrape timing
- Grafana dashboards: corpus throughput, player activity, pipeline health
- Discord webhook alerting via Grafana unified alerting

**Network:**
- All services on `media-network` (Docker external network)
- Container needs outbound HTTPS (port 443) to the CR API
- Dashboard: port 8078, Grafana: port 3000, noVNC: port 6080
