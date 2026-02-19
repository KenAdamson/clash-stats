# CLAUDE.md — Clash Royale Analytics Suite

## Project Owner

Ken (KrylarPrime, tag #L90009GPP). Software engineer at Stoke Space. Polyglot developer — fluent across languages with C# as first love, Python as a strong favorite, and deep experience across the stack. Windows Server administration, home server infrastructure (TrueNAS, Plex, Docker, UniFi), embedded systems (ESP8266), and COM interop by day. Expects professional-grade solutions even in personal projects. Doesn't cut corners.

## What This Is

Two Python tools for analyzing Clash Royale gameplay at a level the community ecosystem doesn't support. These aren't hobbyist scripts — they're the analytics backend for a player who has climbed to 10,900+ trophies with a deck that literally zero other humans play, in ~2,800 lifetime games while peers at the same trophy range have 10,000-15,000+.

The existing community tools (RoyaleAPI, StatsRoyale, Deckshop) are built for meta players running popular decks. They show aggregate stats, not individualized probabilistic analysis. These tools fill that gap.

### `cr_tracker.py` — Battle History Archiver

The CR API only exposes the last 25 battles. No pagination, no historical queries. Community sites only capture games if they happen to poll your tag during the window. This tool polls the API on a schedule, deduplicates via SHA-256 hashing, and archives every battle into SQLite with full card-level data for long-term analysis.

**Intended deployment:** Ken's Plex server (192.168.7.58), which already runs his full media stack via Docker. This should be containerized — Python image, SQLite volume mount for persistent data, cron or scheduler inside the container. Third-party packages are welcome — use the best tool for the job (e.g., `requests` for HTTP, `rich` for terminal output, `pytest` for testing). The RoyaleAPI proxy (proxy.royaleapi.dev) is used to handle dynamic residential IP issues with API key whitelisting.

### `cr_cycle_sim.py` — Push Sequence Monte Carlo Simulator

Models the probability of executing Ken's specific multi-phase push doctrine from any random starting hand. This is NOT a generic cycle calculator — it simulates the actual decision tree of Ken's playstyle across six phases, tracking card positions, cycle costs, and sequence readiness.

## The Deck

```
Card            Elixir  Level   Evo Status
─────────────────────────────────────────────
P.E.K.K.A       7       16      Base (was Evo, swapped out)
Witch            5       16      Evo
Executioner      5       15      Evo (recently swapped IN)
Graveyard        5       15      Base
Miner            3       15      Base
Arrows           3       14      Base
Goblin Curse     2       14      Base
Bats             2       15      Base
─────────────────────────────────────────────
Average Elixir: 4.0
```

**Community data on RoyaleAPI: "Community: 0 — No one else plays."**

This is not a meta deck. It's an 8-year-old core composition that has been continuously refined. The zero community footprint is a strategic asset — opponents at 10,900 have never seen it, can't scout it, and no counter guides exist.

### Evo Slot History (Important for Analytics)

Only two Evo slots are available. Ken recently swapped from Evo Pekka + Evo Witch to **Evo Executioner + Evo Witch**. This swap is backed by data:

- **Evo Pekka era:** 9W-11L (45.0%) over 20 ladder games
- **Evo Executioner era:** 4W-1L (80.0%) over 5 ladder games (and climbing)

The tracker's `_generate_deck_hash()` currently sorts card names with MD5 but **does not include evolution level**, meaning both eras hash to the same deck. This needs to be fixed to track the Evo swap impact over time.

## The Push Doctrine (What the Sim Models)

```
Phase 1: Cycle to Pekka + Witch both in hand (≤5 elixir budget)
Phase 2: Deploy Pekka (7 elixir) — tank initiates push
Phase 3: Deploy Witch (5 elixir) — skeleton screen + splash
Phase 4: Miner + Goblin Curse follow-up — off-lane pressure + conversion
Phase 5: Bats + Graveyard second wave — finisher
Phase 6: Pekka + Witch cycling back — sustain pressure
```

Key Monte Carlo findings (100K trials):
- 49% chance of getting Pekka+Witch ready within 5-elixir budget
- Graveyard is in hand 100% of the time after Phase 4 (deck architecture guarantees finisher)
- 25.4% "smooth sequence" rate (all phases chain with minimal cycling)
- Bats and Goblin Curse are the primary cycle cards (37% each)

## Architecture Decisions

### Use the best packages available
Third-party dependencies are fine everywhere — tracker, sim, analytics, dashboard. Use `requests` over `urllib`, `rich` for terminal output, `pytest` for testing, `pandas` for data analysis, etc. The Docker image handles `pip install`. Pick the best tool for the job.

### SQLite as the data store
Single-file database, portable, zero-config. The `raw_json` column in the battles table preserves the complete API response so no data is lost even if the schema evolves. This is the right call — don't change it.

### Deduplication via content hashing
The API returns overlapping windows (last 25 battles). The SHA-256 hash on `battleTime + tags + crowns` handles dedup cleanly. Don't switch to timestamp-only dedup — the current approach is more robust.

### The sim is pure computation
`cr_cycle_sim.py` has zero I/O dependencies. It's a self-contained Monte Carlo engine. Keep it that way. If we add data-driven features (e.g., weight starting hands by actual battle data), feed the data in via function parameters, don't bolt a database connection onto the sim.

## Known Issues and Improvements

### cr_tracker.py

1. **Deck hash doesn't include Evo status.** The `_generate_deck_hash()` method only hashes card names. It needs to incorporate evolution level so the Evo Pekka and Evo Executioner eras show as distinct decks in `--deck-stats`. This is the highest-priority fix.

2. **No Evo/Hero tracking in deck_cards table.** The `deck_cards` table stores `card_name`, `card_level`, `card_max_level`, `card_elixir`, but not `evolution_level`, `star_level`, or Hero status. The raw JSON has this data — the schema should capture it.

3. **No trophy progression tracking.** We have `player_starting_trophies` and `player_trophy_change` per battle, but no dedicated view for trophy over time. A `--trophy-history` command showing the climb/fall graph (even ASCII) would be valuable.

4. **No streak detection.** Win streaks and loss streaks (especially tilt sessions like the Feb 14 disaster: 15 games, 6-9, -86 trophies) are invisible in the current analytics. Detect and report streaks, including start/end trophies and duration.

5. **No rolling window stats.** RoyaleAPI shows a 35-game rolling window. The tracker should support `--rolling N` to show win rate over the last N games, mimicking and extending what RoyaleAPI provides.

6. **No opponent archetype clustering.** The `--matchups` command shows individual card win rates, but doesn't cluster opponent decks into archetypes (Golem beatdown, Hog cycle, bridge spam, etc.). Even a simple heuristic based on win condition cards would help.

7. **Elixir leak data isn't captured.** The API returns `elixirLeaked` for both players — a strong signal of gameplay efficiency. The battles table should store this. Ken's avg leak in wins is 0.51 vs 1.36 in losses.

8. **No game duration tracking.** Battle time (start to finish) isn't captured. The API's `battleTime` is a timestamp, not duration, but duration can be inferred from tower HP states and crown counts, or from the raw JSON if the API includes it.

9. **Player snapshot diffing.** The `player_snapshots` table captures profile state over time but there's no command to diff snapshots (e.g., "gained 47 trophies, 3 wins, 1 loss since last fetch"). This would make each `--fetch` more informative.

10. **No export capability.** Add `--export csv` and `--export json` for feeding data into external tools or spreadsheets.

### cr_cycle_sim.py

1. **Doesn't model Evo Executioner's tactical impact.** The sim treats all cards as elixir costs and positions. The Evo Executioner swap fundamentally changes push dynamics — 780 damage + knockback means threats that previously reached Pekka now don't. This could be modeled as a "push survival probability" modifier per phase.

2. **Arrows elixir cost is wrong in the DECK dict.** The deck definition has `'Arrows': 3` but Arrows costs 3 elixir in-game so this is actually correct. Verify against current game data if card costs have changed.

3. **No opponent interaction model.** The sim models your cycle in a vacuum. A more advanced version could model common opponent responses (e.g., "opponent drops E-Barbs at bridge during Phase 2") and calculate how the sequence adapts. This is a big lift but would be the ultimate tool.

4. **No integration with tracker data.** The sim uses hardcoded deck composition. It could read the actual deck from the tracker's SQLite database to stay current if cards are swapped.

5. **Single-push analysis only.** The sim models one push sequence. In practice, games involve 2-3 pushes minimum. Modeling the full-game cycle (first push, defend, second push) would give more realistic win probability estimates.

6. **No "what if" mode.** Can't easily test alternate deck compositions (e.g., "what if I swap Arrows for Zap?") without editing the DECK dict. A CLI `--deck` parameter or interactive mode would help with theory-crafting.

## Coding Standards

- **Python 3.8+ compatible.** Although the Docker image will use 3.11, keep backward compat reasonable. Don't use 3.10+ features (match statements, etc.) without justification.
- **Type hints everywhere.** Type hints aren't optional — they're documentation that the interpreter can validate.
- **Docstrings on all public methods.** Google style.
- **No silent failures.** If an API call fails or data is malformed, log it clearly. Ken runs this unattended in Docker — he needs to know when something breaks via `docker logs`.
- **Conservative error handling.** Better to skip a malformed battle and log a warning than crash the whole fetch cycle.
- **Tests welcome.** Especially for the sim's probability calculations — Monte Carlo results should be validated against analytical solutions where possible (e.g., the 21.4% both-in-starting-hand probability is C(6,2)/C(8,4) = 15/70 = 21.43%, and the sim correctly produces ~21.3%).

## Player Context for Smart Analytics

These data points inform what analytics matter:

- **Trophy range:** 10,900+ current, 10,954 PB, targeting 11,500 for ranked mode unlock
- **Play style:** 1-2 games/day surgical precision, NOT volume grinding. 1.35 games/day lifetime average.
- **Efficiency:** ~2,843 games to 10,900 trophies. Peers at same range: 10,000-15,000+ games.
- **Three-crown rate:** 72.9% lifetime (overwhelmingly wins by destruction, not chip)
- **Problem matchups:** Elite Barbarians (60% opponent win rate), Golem+E-Barbs combo
- **Perfect counters:** Sparky (0% opponent win rate, 0/5), Tesla (0/4), Evo Dart Goblin (0/3)
- **Tilt pattern:** Rare but devastating. Feb 14: 15 games in one session, 6-9, dropped 86 trophies. Not wired for volume — wired for precision.
- **Clan War:** Used as zero-risk experimentation lab. Off-deck experiments go 1-4. Only Miner-based experiment won (muscle memory). Ken is a beatdown player, not a cycle player.

## API Reference

**Base URL:** `https://proxy.royaleapi.dev/v1` (via RoyaleAPI proxy) or `https://api.clashroyale.com/v1` (direct)

**Key endpoints:**
- `GET /players/{tag}` — Player profile (all-time stats, current trophies, clan)
- `GET /players/{tag}/battlelog` — Last 25 battles (the polling target)
- `GET /players/{tag}/upcomingchests` — Chest cycle (not currently used)

**Auth:** Bearer token in Authorization header. Key from developer.clashroyale.com.

**Rate limits:** Be respectful. Polling every 2-4 hours is plenty for 1-2 games/day. The proxy has its own limits.

**Player tag encoding:** Tags start with `#` which must be URL-encoded as `%23`. The current code handles this.

## Future Vision

The endgame is a unified analytics platform where the tracker feeds historical data into the sim, the sim's probability model informs strategy decisions, and the whole thing runs as a Docker container on Ken's Plex server with a lightweight web dashboard. Think personal Moneyball for Clash Royale — data-driven competitive advantage for a player who's already proving that precision beats volume.

Priorities:
1. Fix the Evo tracking gap in the tracker (deck hash + schema)
2. Add streak detection and rolling window stats
3. Build a simple web dashboard (even Flask + Chart.js) for visualizing trophy progression and matchup data
4. Integrate tracker data into the sim for data-driven starting hand analysis
5. Add tilt detection — if the tracker sees 3+ consecutive losses, surface a "you're tilting" warning

## Deployment

**Target host:** 192.168.7.58 (Ken's Plex server, Ubuntu-based, Docker already running full media stack)

### Docker Container Design

```
cr-tracker/
├── Dockerfile
├── docker-compose.yml
├── cr_tracker.py
├── cr_cycle_sim.py
├── data/                  ← Volume mount, persists SQLite DB
│   └── clash_royale_history.db
└── .env                   ← CR_API_KEY, CR_PLAYER_TAG (not committed)
```

**Container requirements:**
- Base image: `python:3.11-slim`
- Dependencies managed via `pyproject.toml`, installed with `pip install .` or `pip install -e .[dev]` in the Dockerfile
- SQLite database lives on a Docker volume mount (`./data:/app/data`) so it survives container rebuilds
- Environment variables via `.env` file: `CR_API_KEY`, `CR_PLAYER_TAG=L90009GPP`
- Cron schedule inside container: poll every 3-4 hours (Ken plays 1-2 games/day, 25-game API window means zero risk of missing games at that interval)
- Container should run `--fetch` on the schedule and keep the DB accessible for ad-hoc `--stats`, `--matchups`, etc. via `docker exec`
- Logging to stdout so Docker's log driver captures it

**Health/monitoring:**
- Log each fetch cycle with timestamp, new battle count, and current trophy count
- If the API returns errors for 3+ consecutive fetches, log a clear warning (don't just silently fail — Ken runs this unattended)
- Consider a simple healthcheck endpoint if a web dashboard is added later

**Network:**
- Container needs outbound HTTPS to `proxy.royaleapi.dev` (port 443)
- If the web dashboard is built, expose on a high port (e.g., 8078) — Ken's network is already managing port assignments across his media stack

**Fits alongside existing stack:** This is a tiny container — sub-50MB image, negligible CPU/RAM, writes a few KB to SQLite per fetch. It won't compete with Plex for resources.

## Files

```
cr_tracker.py      — Battle history archiver + analytics CLI
cr_cycle_sim.py    — Push sequence Monte Carlo simulator
CLAUDE.md          — This file (project specification for Claude Code)
```
