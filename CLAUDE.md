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

### Evo Tracking (Important for Analytics)

Only two Evo slots are available. The tracker's `_generate_deck_hash()` currently sorts card names with MD5 but **does not include evolution level**, meaning different Evo configurations hash to the same deck. This needs to be fixed to track Evo swap impact over time.

## Architecture Decisions

### Use the best packages available
Third-party dependencies are fine everywhere — tracker, analytics, dashboard. Use `requests` over `urllib`, `rich` for terminal output, `pytest` for testing, `pandas` for data analysis, etc. The Docker image handles `pip install`. Pick the best tool for the job.

### SQLite as the data store
Single-file database, portable, zero-config. The `raw_json` column in the battles table preserves the complete API response so no data is lost even if the schema evolves. This is the right call — don't change it.

### Deduplication via content hashing
The API returns overlapping windows (last 25 battles). The SHA-256 hash on `battleTime + tags + crowns` handles dedup cleanly. Don't switch to timestamp-only dedup — the current approach is more robust.

## Known Issues and Improvements

### cr_tracker.py

1. **Deck hash doesn't include Evo status.** The `_generate_deck_hash()` method only hashes card names. It needs to incorporate evolution level so different Evo configurations show as distinct decks in `--deck-stats`. This is the highest-priority fix.

2. **No Evo/Hero tracking in deck_cards table.** The `deck_cards` table stores `card_name`, `card_level`, `card_max_level`, `card_elixir`, but not `evolution_level`, `star_level`, or Hero status. The raw JSON has this data — the schema should capture it.

3. **No trophy progression tracking.** We have `player_starting_trophies` and `player_trophy_change` per battle, but no dedicated view for trophy over time. A `--trophy-history` command showing the climb/fall graph (even ASCII) would be valuable.

4. **No streak detection.** Win streaks and loss streaks are invisible in the current analytics. Detect and report streaks, including start/end trophies and duration.

5. **No rolling window stats.** RoyaleAPI shows a 35-game rolling window. The tracker should support `--rolling N` to show win rate over the last N games, mimicking and extending what RoyaleAPI provides.

6. **No opponent archetype clustering.** The `--matchups` command shows individual card win rates, but doesn't cluster opponent decks into archetypes (Golem beatdown, Hog cycle, bridge spam, etc.). Even a simple heuristic based on win condition cards would help.

7. **Elixir leak data isn't captured.** The API returns `elixirLeaked` for both players — a strong signal of gameplay efficiency. The battles table should store this.

8. **No game duration tracking.** Battle time (start to finish) isn't captured. The API's `battleTime` is a timestamp, not duration, but duration can be inferred from tower HP states and crown counts, or from the raw JSON if the API includes it.

9. **Player snapshot diffing.** The `player_snapshots` table captures profile state over time but there's no command to diff snapshots (e.g., "gained 47 trophies, 3 wins, 1 loss since last fetch"). This would make each `--fetch` more informative.

10. **No export capability.** Add `--export csv` and `--export json` for feeding data into external tools or spreadsheets.

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

## Future Vision

The endgame is a unified analytics platform that runs as a Docker container with a lightweight web dashboard. Data-driven competitive advantage for a player who's already proving that precision beats volume.

Priorities:
1. Fix the Evo tracking gap in the tracker (deck hash + schema)
2. Add streak detection and rolling window stats
3. Build a simple web dashboard (even Flask + Chart.js) for visualizing trophy progression and matchup data
4. Add tilt detection — if the tracker sees 3+ consecutive losses, surface a "you're tilting" warning

## Deployment

Docker container designed to run alongside an existing media/services stack.

### Docker Container Design

```
clash-stats/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── src/
│   └── tracker/
│       ├── cr_tracker.py
│       └── test_cr_tracker.py
├── data/                  <- Volume mount, persists SQLite DB
│   └── clash_royale_history.db
└── .env                   <- CR_API_KEY, CR_PLAYER_TAG (not committed)
```

**Container requirements:**
- Base image: `python:3.11-alpine`
- Dependencies managed via `pyproject.toml`, installed with `pip install .` in the Dockerfile
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
