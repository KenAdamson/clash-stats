# CLAUDE.md — Clash Royale Analytics Suite

## Project Overview

A Python tool for analyzing Clash Royale gameplay at a level the community ecosystem doesn't support. This isn't a hobbyist script — it's the analytics backend for a player who has climbed to 10,900+ trophies with a deck that literally zero other humans play, in ~2,800 lifetime games while peers at the same trophy range have 10,000-15,000+.

The existing community tools (RoyaleAPI, StatsRoyale, Deckshop) are built for meta players running popular decks. They show aggregate stats, not individualized probabilistic analysis. This tool fills that gap.

### `cr_tracker.py` — Battle History Archiver

The CR API only exposes the last 25 battles. No pagination, no historical queries. Community sites only capture games if they happen to poll your tag during the window. This tool polls the API on a schedule, deduplicates via SHA-256 hashing, and archives every battle into SQLite with full card-level data for long-term analysis.

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

### SQLite as the data store
Single-file database, portable, zero-config. The `raw_json` column in the battles table preserves the complete API response so no data is lost even if the schema evolves. This is the right call — don't change it.

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

Database migrations are managed by Alembic. The `database.py` module auto-detects pre-Alembic databases and stamps the initial revision. New migrations go in `src/tracker/alembic/versions/`. Run `alembic upgrade head` programmatically on startup via `init_db()`.

## Coding Standards

- **Python 3.11+.** Target the Docker image runtime.
- **Type hints everywhere.** Type hints aren't optional — they're documentation that the interpreter can validate.
- **Docstrings on all public methods.** Google style.
- **No silent failures.** If an API call fails or data is malformed, log it clearly. This runs unattended in Docker — failures must be visible in `docker logs`.
- **Conservative error handling.** Better to skip a malformed battle and log a warning than crash the whole fetch cycle.
- **Tests welcome.**

## Player Context for Smart Analytics

These data points inform what analytics matter:

- **Play style:** 1-2 games/day surgical precision, NOT volume grinding. ~1.35 games/day lifetime average.
- **Efficiency:** ~2,800 games to 10,900+ trophies. Peers at same range: 10,000-15,000+ games.
- **Three-crown rate:** ~73% lifetime (overwhelmingly wins by destruction, not chip)
- **Matchup data matters.** Both problem matchups and hard counters exist — the tracker should surface these from real battle data rather than relying on assumptions.
- **Tilt pattern:** Rare but devastating. Not wired for volume — wired for precision.

## API Reference

**Base URL:** `https://api.clashroyale.com/v1` (direct) or `https://proxy.royaleapi.dev/v1` (via RoyaleAPI proxy, configurable with `CR_API_URL`)

**Key endpoints:**
- `GET /players/{tag}` — Player profile (all-time stats, current trophies, clan)
- `GET /players/{tag}/battlelog` — Last 25 battles (the polling target)
- `GET /players/{tag}/upcomingchests` — Chest cycle (not currently used)

**Auth:** Bearer token in Authorization header. Key from developer.clashroyale.com.

**Rate limits:** Be respectful. Polling every 2-4 hours is plenty for 1-2 games/day. The proxy has its own limits.

**Player tag encoding:** Tags start with `#` which must be URL-encoded as `%23`. The current code handles this.

## ML & Simulation Layer

The replay scraper (`replays.py`) transforms this from a results database into a full game telemetry system. Architecture Decision Records for the ML and simulation capabilities are in [`docs/adr/`](docs/adr/README.md):

- **Monte Carlo Simulation** (ADR-002): Elixir economy modeling, opening hand analysis, Bayesian matchup estimation, card substitution analysis. No ML required — runs immediately on current data.
- **Game State Embeddings** (ADR-003): Phase 0 implemented — 50-dim feature extraction from replay data, two-stage supervised UMAP (50→15-dim analytical, 15→3-dim visualization), HDBSCAN clustering, Euclidean similarity search with percentile rank + Gaussian kernel. Interactive 3D scatter plot (Plotly.js) on the dashboard with click-to-similar. Future phases: TCN-based learned representations.
- **Win Probability Estimator** (ADR-004): P(win) at every game tick. WPA (Win Probability Added) per card placement. Critical play identification.
- **Opponent Prediction** (ADR-005): Sequence model predicting opponent's next card, timing, and position. Markov chain cycle tracking.
- **Counterfactual Simulator** (ADR-006): CVAE generating synthetic game sequences under deck modifications. Deck gradient computation. Manifold-based deck exploration.
- **Training Data Pipeline** (ADR-007): Top-ladder replay corpus for pre-training. Transfer learning to personal games. Meta shift detection.

Dependencies: `torch`, `numpy`, `scikit-learn`, `umap-learn`, `hdbscan`, `pandas` (ML extras installed via `pip install .[ml]` in the Docker image).

## Future Vision

The endgame is a unified analytics platform that runs as a Docker container with a lightweight web dashboard. Data-driven competitive advantage for a player who's already proving that precision beats volume.

Priorities:
1. ~~Fix the Evo tracking gap in the tracker (deck hash + schema)~~ Done
2. ~~Add streak detection and rolling window stats~~ Done
3. ~~Build a simple web dashboard~~ Done (Flask + Chart.js + Plotly.js)
4. ~~Game state embeddings Phase 0 (ADR-003)~~ Done — feature extraction, UMAP, clustering, 3D scatter plot
5. Add tilt detection — if the tracker sees 3+ consecutive losses, surface a "you're tilting" warning
6. Complete BVT on replay scraper pipeline
7. Monte Carlo simulation framework (ADR-002) — first ML milestone, no training data minimum
8. Top-ladder corpus collection (ADR-007) — pre-training data for neural models
9. Game state embeddings Phase 1+ (ADR-003) — TCN-based learned representations, manifold visualization

## Deployment

Docker container designed to run alongside an existing media/services stack.

### Docker Container Design

```
clash-stats/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── docs/
│   └── adr/               ← Architecture Decision Records for ML/simulation layer
│       ├── README.md
│       ├── 001-feature-engineering.md
│       ├── 002-monte-carlo-simulation.md
│       ├── 003-game-state-embeddings.md
│       ├── 004-win-probability-estimator.md
│       ├── 005-opponent-prediction.md
│       ├── 006-counterfactual-simulator.md
│       └── 007-training-data-pipeline.md
├── docker/
│   └── browser/            ← Playwright + NoVNC sidecar for replay scraping
│       ├── Dockerfile
│       └── entrypoint.sh
├── src/
│   └── tracker/
│       ├── __init__.py
│       ├── __main__.py          ← python -m tracker
│       ├── models.py            ← SQLAlchemy ORM (Battle, PlayerSnapshot, DeckCard, ReplayEvent, ReplaySummary)
│       ├── database.py          ← Engine/session setup, Alembic migration runner
│       ├── api.py               ← ClashRoyaleAPI client
│       ├── analytics.py         ← All query/storage functions
│       ├── replays.py           ← RoyaleAPI replay scraper and parser
│       ├── reporting.py         ← Terminal output formatting
│       ├── export.py            ← CSV/JSON export
│       ├── archetypes.py        ← Opponent deck classification
│       ├── cli.py               ← argparse + main() dispatch
│       ├── alembic/             ← Alembic migration config + versions
│       ├── simulation/          ← (planned) Monte Carlo framework (ADR-002)
│       ├── ml/                  ← ML Phase 0: feature extraction, UMAP embeddings, clustering, similarity
│       │   ├── __init__.py
│       │   ├── card_metadata.py ← CardVocabulary — dynamic card→index mapping from DB
│       │   ├── features.py      ← 50-dim feature extraction from replay data
│       │   ├── umap_embeddings.py ← Two-stage UMAP (50→15→3) + supervised fitting
│       │   ├── clustering.py    ← HDBSCAN clustering + cluster profiling
│       │   ├── similarity.py    ← Euclidean distance + percentile rank + Gaussian kernel
│       │   └── storage.py       ← GameFeature/GameEmbedding ORM models, numpy↔BLOB
│       └── tests/
│           ├── conftest.py      ← Shared fixtures
│           ├── fixtures/        ← Static HTML fixtures for replay parser tests
│           ├── test_models.py
│           ├── test_analytics.py
│           ├── test_api.py
│           ├── test_replays.py
│           ├── test_reporting.py
│           ├── test_cli.py
│           ├── test_dashboard.py
│           └── test_export.py
├── data/                  <- Volume mount, persists SQLite DB + feature cache
│   └── clash_royale_history.db
└── .env                   <- CR_API_KEY, CR_PLAYER_TAG (not committed)
```

**Container requirements:**
- Base image: `python:3.11-slim-bookworm`
- Dependencies managed via `pyproject.toml`, installed with `pip install .[ml]` in the Dockerfile
- SQLite database lives on a Docker volume mount (`./data:/app/data`) so it survives container rebuilds
- Environment variables via `.env` file: `CR_API_KEY`, `CR_PLAYER_TAG`
- BusyBox crond schedule: poll every 4 hours (1-2 games/day play rate, 25-game API window means zero risk of missing games)
- Container runs `--fetch` on schedule and keeps the DB accessible for ad-hoc analytics via `docker exec`
- Logging to stdout so Docker's log driver captures it

**Health/monitoring:**
- Log each fetch cycle with timestamp, new battle count, and current trophy count
- If the API returns errors for 3+ consecutive fetches, log a clear warning (don't silently fail — this runs unattended)

**Network:**
- Container needs outbound HTTPS (port 443) to the configured API URL
- Port 8078 reserved for future web dashboard
