# Observability & Resilience Stack

## Architecture

```
┌──────────────────┐  ┌──────────────┐
│    cr-tracker     │  │  cr-browser   │
│  Flask :8078      │  │  Playwright   │
│   /metrics ───────┼──│  noVNC :6080  │
│  cron batch jobs  │  │  CDP :9223    │
└────────┬──────────┘  └──────┬────────┘
         │                    │
    scrape :8078         Docker logs
         │                    │
┌────────┴──────────┐  ┌──────┴────────┐
│    prometheus      │  │    alloy       │
│    :9090           │  │  (log shipper) │
│  15s scrape cycle  │  └──────┬────────┘
└────────┬──────────┘         │ push
         │              ┌──────┴────────┐
         │              │     loki       │
         │              │    :3100       │
         │              │  30d retention │
         │              └──────┬────────┘
         │                     │
    ┌────┴─────────────────────┴────┐
    │           grafana              │
    │           :3000                │
    │  Dashboards + Logs + Alerts   │
    └────────────────────────────────┘
```

All containers share the `media-network` Docker bridge. Only Grafana (:3000) and cr-tracker (:8078) expose host ports. Prometheus, Loki, and Alloy are internal-only.

## Containers

| Container | Image | Purpose | Persistent Data |
|-----------|-------|---------|-----------------|
| cr-tracker | clash-stats-cr-tracker | Pipeline + Flask dashboard + metrics | `./data` volume (SQLite, metrics JSON) |
| cr-browser | clash-stats-cr-browser | Playwright browser for replay scraping | `./data` volume (shared) |
| prometheus | prom/prometheus:latest | Scrapes cr-tracker:8078/metrics every 15s | `prometheus_data` named volume |
| loki | grafana/loki:latest | Log aggregation, 30-day retention | `loki_data` named volume |
| alloy | grafana/alloy:latest | Discovers Docker containers, ships logs to Loki | None (stateless) |
| grafana | grafana/grafana:latest | Dashboards, log viewer, alerting | `grafana_data` named volume |

## Config Files

```
docker/monitoring/
├── prometheus.yml                              # Scrape target: cr-tracker:8078
├── loki-config.yml                             # TSDB store, 30d retention, filesystem backend
├── alloy-config.alloy                          # Docker log discovery → Loki push
└── grafana/
    └── provisioning/
        ├── datasources/
        │   └── datasources.yml                 # Auto-configures Prometheus + Loki
        └── dashboards/
            ├── dashboards.yml                  # Auto-provisioning from /etc/grafana/provisioning/dashboards
            └── pipeline.json                   # Pre-built "Clash Royale Pipeline" dashboard
```

### Prometheus (`prometheus.yml`)

Single scrape target. Prometheus pulls from the Flask app's `/metrics` endpoint, not a separate metrics server.

```yaml
scrape_configs:
  - job_name: "cr-tracker"
    static_configs:
      - targets: ["cr-tracker:8078"]
```

### Loki (`loki-config.yml`)

Filesystem-backed TSDB with 30-day retention. Key config:

- `schema: v13` with `store: tsdb` (current Loki default)
- `retention_period: 30d` with `retention_enabled: true`
- `delete_request_store: filesystem` (required when retention is enabled in recent Loki versions)
- `replication_factor: 1` with `inmemory` kvstore (single-node)

### Alloy (`alloy-config.alloy`)

Discovers all Docker containers via socket, relabels `__meta_docker_container_name` → `container` label, ships logs to Loki. Picks up **all** containers on the host, not just the clash-stats stack — the media stack (jackett, radarr, etc.) logs are also available in Grafana.

### Grafana Provisioning

**Datasources** are auto-configured with hardcoded UIDs:
- Prometheus: `PBFA97CFB590B2093` (default datasource)
- Loki: `P8E80F9AEF21F6940`

These UIDs are referenced in the dashboard JSON. If you recreate datasources manually in the Grafana UI, the dashboard panels will break unless the UIDs match.

**Dashboard** (`pipeline.json`) is auto-provisioned on startup. Grafana re-reads it every 30s (`updateIntervalSeconds: 30` in `dashboards.yml`). To update the dashboard: edit the JSON file and restart Grafana.

## Metrics Architecture

### The Batch Process Problem

The pipeline's scrape jobs (corpus battles, corpus replays, personal fetch) run as short-lived CLI processes via cron. `prometheus_client` counters are in-process memory — they vanish when the process exits. The long-running Flask dashboard serves the `/metrics` endpoint, but it never runs scrape jobs.

### Solution: JSON Accumulator

```
┌─────────────────┐         ┌─────────────────────────┐
│  Batch CLI job   │         │  Flask dashboard (8078)  │
│                  │         │                          │
│  increment       │  flush  │   GET /metrics           │
│  counters ───────┼────────►│   ├─ in-process metrics  │
│                  │  JSON   │   └─ accumulated JSON ───┼──► Prometheus
└─────────────────┘  file    └─────────────────────────┘
```

1. Batch jobs use the same `prometheus_client` Counter/Gauge objects as any instrumented app.
2. At the end of each run, `flush_metrics("job_name")` reads `data/metrics/accumulated.json`, adds the current run's increments, and writes it back.
3. The Flask `/metrics` endpoint serves `generate_latest()` (process-level metrics like Python GC) **plus** the contents of `accumulated.json` rendered as Prometheus text format.
4. `filter_in_process_metrics()` strips batch-only metric names from `generate_latest()` to prevent duplicates (the in-process versions are always 0 since Flask never scrapes).

**Accumulator file location:** `/app/data/metrics/accumulated.json` (on the Docker volume, survives restarts).

**Concurrency:** The cron schedule avoids overlapping batch jobs. If two jobs did overlap, the read-modify-write on the JSON file could lose increments. This is acceptable — the schedule prevents it in practice.

### Metric Definitions

All defined in `src/tracker/metrics.py`.

#### Counters (monotonically increasing)

| Metric | Labels | Incremented By |
|--------|--------|----------------|
| `battles_scraped_total` | `corpus` (personal, top_ladder) | `corpus_scraper.py`, `analytics.py` |
| `battles_deduped_total` | `corpus` | (defined, not yet instrumented) |
| `replays_fetched_total` | `source` (scraper) | `replays.py` |
| `replays_failed_total` | `error_type` (transient, auth_expired, cloudflare, parse_error, no_events) | `replays.py` |
| `api_requests_total` | `endpoint`, `status` | `api.py` on every HTTP request |
| `session_expiry_total` | — | `replays.py` on RoyaleAPI auth redirect |
| `corpus_players_deactivated_total` | — | `corpus_scraper.py` on 404 |
| `circuit_breaker_trips_total` | `breaker` | `replays.py` when circuit opens |
| `scrape_runs_total` | `scrape_type` (battles, replays), `outcome` (success, partial, failed) | `corpus_scraper.py` at end of each run |

#### Gauges (current value)

| Metric | Set By |
|--------|--------|
| `corpus_players_active` | `corpus_scraper.py` — queries `PlayerCorpus.active == 1` count at start of each battle scrape |

#### Histograms

| Metric | Labels | Buckets |
|--------|--------|---------|
| `api_request_duration_seconds` | `endpoint` | 0.5, 1.0, 2.0, 5.0, 10.0, 30.0 |

### _created Series Suppression

`prometheus_client` emits `_created` companion timestamps for every counter by default. These cause duplicate entries in Grafana stat panels. Suppressed via `PROMETHEUS_DISABLE_CREATED_SERIES=true` set in `cli.py` and `dashboard.py` before any prometheus_client import.

## Grafana Dashboard

**URL:** `http://localhost:3000/d/cr-pipeline/clash-royale-pipeline`
**Login:** admin / `$GRAFANA_PASSWORD` (default: `changeme`)

### Panel Layout

| Row | Panel | Type | Query |
|-----|-------|------|-------|
| 0 | Battles Scraped | Bar chart timeseries | `increase(battles_scraped_total[10m])` |
| 0 | Total Battles | Stat | `sum(battles_scraped_total)` |
| 8 | Replays Fetched vs Failed | Timeseries | `increase(replays_fetched_total[1h])` / `increase(replays_failed_total[1h])` |
| 16 | API Request Rate | Timeseries | `rate(api_requests_total[5m])` |
| 16 | API Latency (p95) | Timeseries | `histogram_quantile(0.95, rate(...[5m]))` |
| 16 | Pipeline Health | Stat | `corpus_players_active`, `session_expiry_total`, `circuit_breaker_trips_total` |
| 24 | Scrape Runs (24h) | Stat (stoplight) | `increase(scrape_runs_total{outcome=...}[24h])` — green/yellow/red |
| 24 | Container Logs | Logs (Loki) | `{container=~"cr-tracker\|cr-browser"}` |

### Prometheus `increase()` Quirk

`increase()` on counters can return fractional values (e.g., 2.02 instead of 2) due to Prometheus extrapolating at scrape interval boundaries. Panels that display counts use `"decimals": 0` in their fieldConfig to round the display.

## Resilience

### Exception Hierarchy (`api.py`)

```
APIError
├── RateLimitError    (429)
├── AuthError         (401, 403)
├── NotFoundError     (404)
├── ServerError       (5xx)
└── ConnectionError_  (network/timeout)
```

### Retry (`api.py`)

All CR API calls go through `_request()` which is decorated with `tenacity`:

- **Retries on:** RateLimitError, ServerError, ConnectionError_
- **Does NOT retry:** AuthError, NotFoundError (permanent failures)
- **Strategy:** Exponential backoff with jitter — ~2s, ~4s, ~8s + up to 3s random
- **Max attempts:** 4
- **Logging:** Each retry logged at WARNING level via `before_sleep_log`

### Circuit Breaker (`replays.py`)

`_navigate_authenticated()` is wrapped with `@circuit`:

- **Failure threshold:** 3 consecutive `SessionExpiredError` exceptions
- **Recovery timeout:** 300 seconds (5 minutes)
- **Effect:** When open, all replay fetches for all players fail-fast without hitting RoyaleAPI

The corpus replay scraper also has an application-level circuit breaker: 3 consecutive players returning -1 (session expired) stops the entire run.

### Error Handling in Corpus Scraper

| Error | Action |
|-------|--------|
| `NotFoundError` (404) | Deactivate player (`active = 0`), never retry |
| `AuthError` (401/403) | Stop entire run — API key is bad |
| `RateLimitError` / `ServerError` / `ConnectionError_` | Skip player, increment transient counter, retry next run |
| Session expired (replays) | After 3 consecutive, stop replay run early |

## Troubleshooting

### Container won't start

Check logs: `docker logs <container> --tail 50`

Common issues:
- **Loki crash loop:** Config validation failure. Recent Loki versions require `delete_request_store` when `retention_enabled: true`.
- **Grafana datasource errors:** UID mismatch between `datasources.yml` and `pipeline.json`. The UIDs must match exactly.
- **cr-tracker crash on startup:** The initial `--fetch` in `entrypoint.sh` runs before cron starts. If it fails (bad API key, network), the container exits.

### Dashboard shows "No data"

1. **Check Prometheus target:** `docker exec grafana wget -qO- 'http://prometheus:9090/api/v1/targets'` — the cr-tracker target should show `"health": "up"`.
2. **Check /metrics endpoint:** `docker exec cr-tracker curl -sf http://localhost:8078/metrics | grep battles_scraped` — should show accumulated values.
3. **Check accumulated JSON:** `docker exec cr-tracker cat /app/data/metrics/accumulated.json` — should have non-zero values after at least one scrape run.
4. **Run a scrape to generate data:** `docker exec cr-tracker clash-stats --corpus-scrape --corpus-limit 3 --db /app/data/clash_royale_history.db`

### Metrics showing wrong values

**Spike larger than total:** The `/metrics` endpoint was serving duplicate metric lines (one from the in-process registry at 0, one from the accumulated JSON with real values). Prometheus counter reset detection interprets the 0→N jump as a new increase on top of previous values. Fixed by `filter_in_process_metrics()` which strips batch-only metric names from `generate_latest()`. If it recurs, check `docker exec cr-tracker curl -sf http://localhost:8078/metrics | grep -c <metric_name>` — should be exactly 1 value line per label combination.

**Fractional counts (e.g. 2.02):** Normal Prometheus `increase()` behavior. Set `"decimals": 0` in the Grafana panel fieldConfig.

### Loki not receiving logs

1. **Check Alloy:** `docker logs alloy --tail 20` — should show discovery of Docker containers.
2. **Check Loki labels:** `docker exec grafana wget -qO- 'http://loki:3100/loki/api/v1/label/container/values'` — should list all running containers.
3. **Verify Docker socket:** Alloy mounts `/var/run/docker.sock:ro`. If Alloy can't access it, no logs are discovered.

### Resetting metrics

Delete the accumulator and restart: `docker exec cr-tracker rm /app/data/metrics/accumulated.json && docker compose restart cr-tracker`. Prometheus will see a counter reset and handle it gracefully (the old data stays in Prometheus's TSDB with correct historical values).

### Adding new metrics

1. Define the Counter/Gauge/Histogram in `metrics.py`.
2. Add the base name (without `_total`/`_bucket` suffix) to `BATCH_METRIC_NAMES` in `metrics.py`.
3. Increment it in the relevant module.
4. If it's in a batch CLI command, `flush_metrics()` picks it up automatically.
5. Add a panel to `pipeline.json` and restart Grafana.

## Environment Variables

| Variable | Default | Used By |
|----------|---------|---------|
| `GRAFANA_PASSWORD` | `changeme` | Grafana admin password |
| `REPLAYS_PER_PLAYER` | `25` | Max replays fetched per player per run |
| `PROMETHEUS_METRICS_FILE` | `/app/data/metrics/accumulated.json` | Metrics accumulator location |
| `PROMETHEUS_DISABLE_CREATED_SERIES` | `true` | Suppress `_created` companion metrics |

## Rebuilding from Scratch

```bash
# 1. Bring up the full stack
docker compose up -d

# 2. Verify all containers are running
docker compose ps

# 3. Check Prometheus is scraping
docker exec grafana wget -qO- 'http://prometheus:9090/api/v1/targets' | python3 -m json.tool

# 4. Check Loki has logs
docker exec grafana wget -qO- 'http://loki:3100/loki/api/v1/label/container/values'

# 5. Run a scrape to seed metrics
docker exec cr-tracker clash-stats --corpus-scrape --corpus-limit 5 --db /app/data/clash_royale_history.db

# 6. Open Grafana
# http://localhost:3000 — admin / $GRAFANA_PASSWORD
# Dashboard: Clash Royale Pipeline (auto-provisioned)
```

Named volumes (`prometheus_data`, `loki_data`, `grafana_data`) persist across `docker compose down` / `up` cycles. To fully reset observability data: `docker compose down -v` (destroys volumes).
