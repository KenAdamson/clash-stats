"""Prometheus metrics for the Clash Royale pipeline.

Centralized metric definitions. Two modes:

1. **Long-running** (dashboard): the Flask app serves /metrics combining
   in-process metrics with accumulated batch job metrics from a JSON file.
2. **Batch** (CLI cron jobs): increment counters, then call flush_metrics()
   to persist accumulated totals to a shared JSON file.

This solves the problem of cron jobs being short-lived processes whose
in-memory counters disappear when the process exits.
"""

import json
import logging
import os
from pathlib import Path

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

logger = logging.getLogger(__name__)

# Shared metrics accumulator file — persisted on Docker volume
METRICS_FILE = Path(os.environ.get("PROMETHEUS_METRICS_FILE", "/app/data/metrics/accumulated.json"))
TEXTFILE_DIR = METRICS_FILE.parent

# ---------------------------------------------------------------------------
# Battle scraping
# ---------------------------------------------------------------------------

BATTLES_SCRAPED = Counter(
    "battles_scraped_total",
    "Battles written to DB",
    ["corpus"],  # personal, top_ladder
)

BATTLES_DEDUPED = Counter(
    "battles_deduped_total",
    "Battles skipped (already in DB)",
    ["corpus"],
)

# ---------------------------------------------------------------------------
# Replay scraping
# ---------------------------------------------------------------------------

REPLAYS_FETCHED = Counter(
    "replays_fetched_total",
    "Replays successfully fetched and stored",
    ["source"],  # personal, corpus
)

REPLAYS_FAILED = Counter(
    "replays_failed_total",
    "Replay fetch failures",
    ["error_type"],  # transient, auth_expired, cloudflare, parse_error, no_events
)

# ---------------------------------------------------------------------------
# CR API
# ---------------------------------------------------------------------------

API_REQUESTS = Counter(
    "api_requests_total",
    "CR API requests",
    ["endpoint", "status"],  # endpoint=players|battlelog|leaderboard, status=200|429|etc
)

API_LATENCY = Histogram(
    "api_request_duration_seconds",
    "CR API call duration",
    ["endpoint"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

# ---------------------------------------------------------------------------
# Pipeline health
# ---------------------------------------------------------------------------

SESSION_EXPIRY = Counter(
    "session_expiry_total",
    "RoyaleAPI session expiry events",
)

CORPUS_PLAYERS_ACTIVE = Gauge(
    "corpus_players_active",
    "Number of active corpus players",
)

CORPUS_PLAYERS_DEACTIVATED = Counter(
    "corpus_players_deactivated_total",
    "Players deactivated (404 / permanently failed)",
)

CIRCUIT_BREAKER_TRIPS = Counter(
    "circuit_breaker_trips_total",
    "Circuit breaker trip events",
    ["breaker"],  # royaleapi_auth, cr_api
)

SCRAPE_RUNS = Counter(
    "scrape_runs_total",
    "Scrape run completions",
    ["scrape_type", "outcome"],  # type=battles|replays, outcome=success|partial|failed
)


def _read_accumulated() -> dict:
    """Read accumulated metrics from the shared JSON file."""
    try:
        if METRICS_FILE.exists():
            return json.loads(METRICS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _write_accumulated(data: dict) -> None:
    """Write accumulated metrics to the shared JSON file."""
    TEXTFILE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps(data, indent=2))


def flush_metrics(job_name: str = "batch") -> None:
    """Persist current process counter values to a shared accumulator.

    Reads existing accumulated totals, adds this run's increments,
    and writes back. The dashboard's /metrics endpoint renders these.

    Args:
        job_name: Label for logging.
    """
    from prometheus_client import REGISTRY

    accumulated = _read_accumulated()

    for metric in REGISTRY.collect():
        for sample in metric.samples:
            # Skip internal python_* and process_* metrics
            if sample.name.startswith(("python_", "process_")):
                continue
            # Skip _created and _bucket suffix samples for simplicity
            if sample.name.endswith(("_created", "_info")):
                continue

            key = sample.name
            if sample.labels:
                label_str = ",".join(f'{k}="{v}"' for k, v in sorted(sample.labels.items()))
                key = f"{sample.name}{{{label_str}}}"

            if sample.value and sample.value > 0:
                if sample.name.endswith("_bucket"):
                    # Histogram buckets: accumulate
                    accumulated[key] = accumulated.get(key, 0) + sample.value
                elif "gauge" in metric.type:
                    # Gauges: overwrite (latest value wins)
                    accumulated[key] = sample.value
                else:
                    # Counters: accumulate
                    accumulated[key] = accumulated.get(key, 0) + sample.value

    _write_accumulated(accumulated)
    logger.info("Metrics accumulated to %s (%s)", METRICS_FILE, job_name)


# Metric names that are only meaningful from batch jobs.
# These must be stripped from generate_latest() to avoid duplicates
# with the accumulated JSON values.
BATCH_METRIC_NAMES = {
    "battles_scraped",
    "battles_deduped",
    "replays_fetched",
    "replays_failed",
    "api_requests",
    "api_request_duration_seconds",
    "session_expiry",
    "corpus_players_active",
    "corpus_players_deactivated",
    "circuit_breaker_trips",
    "scrape_runs",
}


def render_accumulated_metrics() -> str:
    """Render accumulated batch metrics in Prometheus text format.

    Called by the dashboard's /metrics endpoint.
    """
    accumulated = _read_accumulated()
    if not accumulated:
        return ""

    lines = []
    for key, value in sorted(accumulated.items()):
        lines.append(f"{key} {value}")
    return "\n".join(lines) + "\n"


def filter_in_process_metrics(raw: str) -> str:
    """Remove batch-only metrics from generate_latest() output.

    Prevents duplicate series when the accumulated JSON also provides
    these metrics with real values from batch jobs.
    """
    lines = raw.split("\n")
    filtered = []
    skip = False
    for line in lines:
        if line.startswith("# HELP ") or line.startswith("# TYPE "):
            metric_name = line.split()[2]
            skip = metric_name in BATCH_METRIC_NAMES
        elif not line.startswith("#") and line.strip():
            metric_name = line.split("{")[0].split()[0]
            # Strip _total, _bucket, _count, _sum suffixes to get base name
            base = metric_name
            for suffix in ("_total", "_bucket", "_count", "_sum"):
                if base.endswith(suffix):
                    base = base[:-len(suffix)]
                    break
            skip = base in BATCH_METRIC_NAMES

        if not skip:
            filtered.append(line)

    return "\n".join(filtered)
