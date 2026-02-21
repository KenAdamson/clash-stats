# Clash Stats

Analytics suite for Clash Royale — battle tracker and push sequence simulator.

Built for players who care about data, not just meta. The CR API only exposes the last 25 battles with no historical queries. Community tools only capture games if they happen to poll during that window. This fills the gap: poll on a schedule, deduplicate, archive everything, and surface analytics that RoyaleAPI/StatsRoyale/Deckshop don't provide.

## Components

### Battle Tracker (`cr_tracker.py`)

Polls the CR API, deduplicates via SHA-256 hashing, and archives every battle into SQLite with full card-level data.

```
python src/tracker/cr_tracker.py --fetch --api-key YOUR_KEY --player-tag L90009GPP
python src/tracker/cr_tracker.py --stats
python src/tracker/cr_tracker.py --deck-stats
python src/tracker/cr_tracker.py --matchups
python src/tracker/cr_tracker.py --crowns
python src/tracker/cr_tracker.py --recent 20
```

### Push Sequence Simulator (`cr_cycle_sim.py`)

Monte Carlo simulator that models the probability of executing a specific multi-phase push doctrine from any random starting hand. Not a generic cycle calculator — it simulates actual decision trees across six phases, tracking card positions, cycle costs, and sequence readiness.

## Setup

### Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Environment Variables

```bash
cp .env.example .env
# Edit .env with your API key and player tag
```

| Variable | Required | Description |
|---|---|---|
| `CR_API_KEY` | Yes | API key from [developer.clashroyale.com](https://developer.clashroyale.com) |
| `CR_PLAYER_TAG` | Yes | Player tag without `#` |
| `CR_API_URL` | No | API base URL (default: `https://api.clashroyale.com/v1`) |

To use the RoyaleAPI proxy (avoids IP whitelisting issues with dynamic IPs):

```
CR_API_URL=https://proxy.royaleapi.dev/v1
```

## Docker

Alpine-based image with BusyBox crond. Fetches every 4 hours, logs to stdout.

```bash
cp .env.example .env
# Edit .env with your credentials

docker compose up -d
docker logs -f cr-tracker
```

Ad-hoc analytics:

```bash
docker exec cr-tracker python cr_tracker.py --stats --db /app/data/clash_royale_history.db
docker exec cr-tracker python cr_tracker.py --matchups --db /app/data/clash_royale_history.db
```

The SQLite database persists on a volume mount at `./data/`.

## Tests

```bash
pytest
```

57 tests covering the database layer, API client, analytics queries, reporting, CLI, and the full fetch pipeline.

## Project Structure

```
├── src/
│   ├── tracker/
│   │   ├── cr_tracker.py        # Battle history archiver + analytics CLI
│   │   └── test_cr_tracker.py   # Test suite
│   └── cycle_sim/
│       └── cr_cycle_sim.py      # Push sequence Monte Carlo simulator
├── Dockerfile                   # Alpine + BusyBox crond
├── docker-compose.yml
├── entrypoint.sh
├── crontab
├── pyproject.toml
└── .env.example
```
