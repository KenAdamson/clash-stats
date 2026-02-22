# ADR-008: Pipeline Observability & Resilience Infrastructure

**Status:** Implemented
**Date:** 2026-02-22
**Depends on:** ADR-007 (Training Data Pipeline)

## Context

The corpus pipeline (200 players, scraping battles every 6 hours, replays every 6 hours) had zero resilience and no observability:

- **Logging was dead in production.** No `logging.basicConfig()` anywhere — all `logger.*` calls in `replays.py` and `corpus_scraper.py` silently discarded.
- **`api.py` raised untyped `Exception`.** Callers couldn't distinguish 429 from 401 from 404 from 503.
- **No retry logic.** Transient failures permanently skipped replays/battles.
- **No circuit breaker.** An expired RoyaleAPI session burned through 200 players getting 0 replays each.
- **Silent swallows.** Cloudflare timeouts, JSON parse failures — all invisible.
- **Permanent failures never penalized.** A deleted player (404) retried every run forever.
- **No metrics or dashboards.** "How many replays succeeded today?" required grepping logs that didn't exist.

For a pipeline running unattended on a home server, this is unacceptable. Failures must be visible, transient errors must be retried, and permanent errors must be handled gracefully.

## Decision

### Resilience: tenacity + circuitbreaker

**Chosen:** `tenacity` for retry with backoff, `circuitbreaker` for circuit breaker pattern.

**Why tenacity:** Python's closest equivalent to .NET Polly. Composable retry strategies with exponential backoff, jitter, per-exception-type policies, and logging hooks. Mature, well-maintained, zero dependencies.

**Why circuitbreaker:** Lightweight, decorator-based, async-native. Exactly what's needed to stop burning through 200 players when the RoyaleAPI session expires.

**Rejected alternatives:**
- `stamina` — opinionated wrapper around tenacity, less control over retry policies
- `pybreaker` — less maintained, no async support
- Manual retry loops — error-prone, no jitter, no composability

### Structured Exception Hierarchy

```
APIError
├── RateLimitError    (429)
├── AuthError         (401, 403)
├── NotFoundError     (404)
├── ServerError       (5xx)
└── ConnectionError_  (network/timeout)
```

Callers handle each class differently:
- **RateLimitError/ServerError/ConnectionError_** → retry with exponential backoff + jitter (4 attempts, 2s→60s)
- **AuthError** → stop entire run (no point continuing with bad credentials)
- **NotFoundError** → deactivate player permanently (deleted account)

### Observability: PLG Stack (Prometheus + Loki + Grafana + Alloy)

**Chosen:** Self-hosted PLG stack via Docker Compose.

| Component | Role | Image |
|---|---|---|
| Prometheus | Metrics scraping & storage | `prom/prometheus:latest` |
| Loki | Log aggregation & querying | `grafana/loki:latest` |
| Alloy | Docker log shipping (replaces Promtail) | `grafana/alloy:latest` |
| Grafana | Dashboards, alerting, log viewer | `grafana/grafana:latest` |

**Why PLG:**
- Industry standard for self-hosted observability
- All components are lightweight enough for a single host
- Grafana provides unified view: metrics + logs + alerts in one UI
- Pre-provisioned dashboards and datasources via config files (no manual setup)
- Alloy is the modern replacement for Promtail with native Docker discovery

**Rejected alternatives:**
- **SigNoz** — full-featured but heavy (ClickHouse backend, 4GB+ RAM baseline)
- **Graylog** — requires MongoDB + Elasticsearch, overkill for single-host
- **Netdata** — excellent for host metrics, no log aggregation
- **Uptime Kuma** — blackbox monitoring only, no application metrics
- **ELK Stack** — Elasticsearch alone needs 2GB+ heap, way too heavy
- **Datadog/New Relic** — SaaS, costs money, sends data off-premise

### Orchestration: Docker Compose (NOT Kubernetes)

**Decision:** Stay on Docker Compose. No Kubernetes.

**Rationale:** This is 7 containers on a single host (cr-tracker, cr-browser, prometheus, loki, alloy, grafana, plus the existing media stack). Docker Compose handles this perfectly with:
- `depends_on` for startup ordering
- `restart: unless-stopped` for crash recovery
- Shared `media-network` bridge for inter-container DNS
- Named volumes for persistent data
- Health checks on key services

Kubernetes adds complexity (etcd, kube-apiserver, kubelet, scheduler, controller-manager) for zero benefit at this scale. The operational overhead of a single-node k8s cluster exceeds the overhead of what it's orchestrating.

## Metrics Schema

### Counters
| Metric | Labels | Description |
|---|---|---|
| `battles_scraped_total` | `corpus` | Battles written to DB |
| `battles_deduped_total` | `corpus` | Battles skipped (already exist) |
| `replays_fetched_total` | `source` | Replays successfully fetched |
| `replays_failed_total` | `error_type` | Replay fetch failures |
| `api_requests_total` | `endpoint`, `status` | CR API requests |
| `session_expiry_total` | — | RoyaleAPI session expiry events |
| `corpus_players_deactivated_total` | — | Players deactivated (404) |
| `circuit_breaker_trips_total` | `breaker` | Circuit breaker trips |
| `scrape_runs_total` | `scrape_type`, `outcome` | Scrape run completions |

### Gauges
| Metric | Description |
|---|---|
| `corpus_players_active` | Active corpus players |

### Histograms
| Metric | Labels | Description |
|---|---|---|
| `api_request_duration_seconds` | `endpoint` | API call latency |

## Grafana Dashboard

Auto-provisioned `pipeline.json` with panels:
- **Battles Scraped** — `increase(battles_scraped_total[1h])` by corpus
- **Replays Fetched vs Failed** — fetched vs failed over time
- **API Request Rate** — `rate(api_requests_total[5m])` by endpoint/status
- **API Latency (p95/p50)** — histogram quantiles
- **Pipeline Health** — stat panel with active players, session expiries, CB trips
- **Scrape Run Outcomes** — success/failure by scrape type
- **Container Logs** — live Loki log panel for cr-tracker and cr-browser

## File Changes

| File | Change |
|---|---|
| `src/tracker/metrics.py` | New — centralized Prometheus metric definitions |
| `src/tracker/api.py` | Structured exceptions, tenacity retry, metrics instrumentation |
| `src/tracker/replays.py` | Circuit breaker, retry, fixed silent swallows, metrics |
| `src/tracker/corpus_scraper.py` | Typed exception handling, player deactivation, metrics |
| `src/tracker/cli.py` | `logging.basicConfig()` for all log output |
| `src/tracker/dashboard.py` | Metrics server startup on port 8001 |
| `pyproject.toml` | Added tenacity, circuitbreaker, prometheus-client |
| `docker-compose.yml` | Added prometheus, loki, alloy, grafana; health checks; volumes |
| `docker/monitoring/` | New — prometheus.yml, loki-config.yml, alloy-config.alloy |
| `docker/monitoring/grafana/` | New — provisioned datasources + dashboard |
| `entrypoint.sh` | Metrics port announcement |

## Consequences

- **Positive:** Pipeline failures are now visible (logs in Loki, metrics in Prometheus, dashboards in Grafana). Transient errors auto-retry. Permanent errors are handled gracefully. Session expiry is circuit-broken.
- **Positive:** Single `docker compose up -d` brings up the entire stack including observability.
- **Negative:** 4 additional containers (~500MB additional RAM). Acceptable on a 31GB host.
- **Negative:** Grafana dashboard UIDs are hardcoded — must match provisioned datasource UIDs. Fragile but standard for provisioned Grafana.
