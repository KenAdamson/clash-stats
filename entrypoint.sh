#!/bin/sh
set -e

# Yield CPU to higher-priority host workloads (e.g. Plex). All children
# (gunicorn, cron, cron-spawned jobs) inherit this nice level.
renice -n 15 $$ >/dev/null 2>&1 || true

# DATABASE_URL is the canonical DB connection string (PostgreSQL).
# --db flag is only used as a fallback label; DATABASE_URL always takes precedence in cli.py.
# If DATABASE_URL is not set, fail fast rather than silently writing to SQLite.
if [ -z "${DATABASE_URL}" ]; then
    echo "FATAL: DATABASE_URL is not set. Refusing to start — PostgreSQL is the only supported backend."
    exit 1
fi
DB_FLAG="--db ${DATABASE_URL}"
LOCKDIR=/tmp/locks
rm -rf "$LOCKDIR"
mkdir -p "$LOCKDIR"

# Cron runs jobs in a CLEAN environment — container env vars (set by compose)
# are NOT visible to cron-spawned jobs. Anything a job needs must be baked into
# its wrapper at startup. CR_API_KEY etc. are baked inline below; the RoyaleAPI
# scraper's proxy/solver/rotation endpoints are baked here into one reusable
# block. Without this, cron replay scraping bypasses the VPN proxy and egresses
# from the container's (banned) residential IP. Each var is included only if
# non-empty — exporting an empty GLUETUN_CONTROL_URL would disable rotation.
SCRAPER_ENV_EXPORTS=""
[ -n "${ROYALEAPI_PROXY}" ]      && SCRAPER_ENV_EXPORTS="${SCRAPER_ENV_EXPORTS}export ROYALEAPI_PROXY=\"${ROYALEAPI_PROXY}\"
"
[ -n "${FLARESOLVERR_URL}" ]     && SCRAPER_ENV_EXPORTS="${SCRAPER_ENV_EXPORTS}export FLARESOLVERR_URL=\"${FLARESOLVERR_URL}\"
"
[ -n "${GLUETUN_CONTROL_URL}" ]  && SCRAPER_ENV_EXPORTS="${SCRAPER_ENV_EXPORTS}export GLUETUN_CONTROL_URL=\"${GLUETUN_CONTROL_URL}\"
"
[ -n "${ROYALEAPI_REQUESTS_PER_SEC}" ] && SCRAPER_ENV_EXPORTS="${SCRAPER_ENV_EXPORTS}export ROYALEAPI_REQUESTS_PER_SEC=\"${ROYALEAPI_REQUESTS_PER_SEC}\"
"

# Build fetch script with baked-in env vars
# (Debian cron runs jobs in a clean environment)
cat > /app/fetch.sh << EOF
#!/bin/sh
exec flock -n ${LOCKDIR}/fetch.lock sh -c '
cd /app
export CR_API_KEY="${CR_API_KEY}"
export CR_PLAYER_TAG="${CR_PLAYER_TAG}"
[ -n "${CR_API_URL}" ] && export CR_API_URL="${CR_API_URL}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --fetch ${DB_FLAG}
' || echo "fetch: previous run still active, skipping"
EOF
chmod +x /app/fetch.sh

# Build personal combined wrapper: fetch battles + replays
cat > /app/personal_combined.sh << EOF
#!/bin/sh
exec flock -n ${LOCKDIR}/personal_combined.lock sh -c '
cd /app
export CR_API_KEY="${CR_API_KEY}"
export CR_PLAYER_TAG="${CR_PLAYER_TAG}"
[ -n "${CR_API_URL}" ] && export CR_API_URL="${CR_API_URL}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
${SCRAPER_ENV_EXPORTS}
export PYTHONUNBUFFERED=1
clash-stats --personal-combined --player-tag "${CR_PLAYER_TAG}" ${DB_FLAG}
' || echo "personal_combined: previous run still active, skipping"
EOF
chmod +x /app/personal_combined.sh

# Alt-account combined wrapper: same battles+replays pass as personal_combined
# but for CR_ALT_TAG, with corpus label 'alt' so alt games never pollute
# main-account analytics (dashboard/tilt/trophy-history all filter on
# corpus='personal'). No-ops cleanly when CR_ALT_TAG is unset — the cron line
# is unconditional, the wrapper decides. CR_PLAYER_TAG is deliberately NOT
# exported (cli falls back to it; --player-tag is explicit here).
cat > /app/alt_combined.sh << EOF
#!/bin/sh
[ -z "${CR_ALT_TAG}" ] && exit 0
exec flock -n ${LOCKDIR}/alt_combined.lock sh -c '
cd /app
export CR_API_KEY="${CR_API_KEY}"
[ -n "${CR_API_URL}" ] && export CR_API_URL="${CR_API_URL}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
${SCRAPER_ENV_EXPORTS}
export PYTHONUNBUFFERED=1
clash-stats --personal-combined --player-tag "${CR_ALT_TAG}" --corpus-label alt ${DB_FLAG}
' || echo "alt_combined: previous run still active, skipping"
EOF
chmod +x /app/alt_combined.sh

# Build corpus wrapper scripts for crond
cat > /app/corpus_update.sh << EOF
#!/bin/sh
exec flock -n ${LOCKDIR}/corpus_update.lock sh -c '
cd /app
export CR_API_KEY="${CR_API_KEY}"
[ -n "${CR_API_URL}" ] && export CR_API_URL="${CR_API_URL}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --corpus-update --corpus-limit 500 ${DB_FLAG}
' || echo "corpus_update: previous run still active, skipping"
EOF
chmod +x /app/corpus_update.sh

cat > /app/corpus_scrape.sh << EOF
#!/bin/sh
exec flock -n ${LOCKDIR}/corpus_scrape.lock sh -c '
cd /app
export CR_API_KEY="${CR_API_KEY}"
[ -n "${CR_API_URL}" ] && export CR_API_URL="${CR_API_URL}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --corpus-scrape --corpus-limit 500 ${DB_FLAG}
' || echo "corpus_scrape: previous run still active, skipping"
EOF
chmod +x /app/corpus_scrape.sh

cat > /app/sim_refresh.sh << EOF
#!/bin/sh
exec flock -n ${LOCKDIR}/sim_refresh.lock sh -c '
cd /app
export CR_PLAYER_TAG="${CR_PLAYER_TAG}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --sim-full --player-tag "${CR_PLAYER_TAG}" ${DB_FLAG}
' || echo "sim_refresh: previous run still active, skipping"
EOF
chmod +x /app/sim_refresh.sh

# Derived dimensions: rebuild clan_dim (CR clan API) + player_dim (from battles).
# Both tables are fully derived/repopulatable — the refresh TRUNCATES and
# rebuilds. flock prevents overlap. NOTE: the matching crontab line is left
# COMMENTED OUT (see crontab) because the rebuild is destructive-by-design;
# enable it deliberately once the migration has run and a manual
# `clash-stats --refresh-dims` has been verified against the live DB.
cat > /app/refresh_dims.sh << EOF
#!/bin/sh
exec flock -n ${LOCKDIR}/refresh_dims.lock sh -c '
cd /app
export CR_API_KEY="${CR_API_KEY}"
[ -n "${CR_API_URL}" ] && export CR_API_URL="${CR_API_URL}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --refresh-dims ${DB_FLAG}
' || echo "refresh_dims: previous run still active, skipping"
EOF
chmod +x /app/refresh_dims.sh

# Weekly corpus hygiene: enrich + deactivate bots and dormant accounts so the
# FIFO scraper re-polls the live core more often (higher games/player density).
cat > /app/prune_corpus.sh << EOF
#!/bin/sh
exec flock -n ${LOCKDIR}/prune_corpus.lock sh -c '
cd /app
export CR_API_KEY="${CR_API_KEY}"
[ -n "${CR_API_URL}" ] && export CR_API_URL="${CR_API_URL}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --prune-corpus ${DB_FLAG}
' || echo "prune_corpus: previous run still active, skipping"
EOF
chmod +x /app/prune_corpus.sh

# (The corpus_replays.sh wrapper is defined further below — the legacy
# Playwright-based version that lived here was dead code, overwritten by the
# HTTP-path version at write time.)

# Network discovery: mine opponent tags and add to corpus
cat > /app/corpus_discover.sh << EOF
#!/bin/sh
exec flock -n ${LOCKDIR}/corpus_discover.lock sh -c '
cd /app
export CR_API_KEY="${CR_API_KEY}"
[ -n "${CR_API_URL}" ] && export CR_API_URL="${CR_API_URL}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --corpus-discover --corpus-limit 500 ${DB_FLAG}
' || echo "corpus_discover: previous run still active, skipping"
EOF
chmod +x /app/corpus_discover.sh

# Location leaderboard discovery
cat > /app/corpus_locations.sh << EOF
#!/bin/sh
exec flock -n ${LOCKDIR}/corpus_locations.lock sh -c '
cd /app
export CR_API_KEY="${CR_API_KEY}"
[ -n "${CR_API_URL}" ] && export CR_API_URL="${CR_API_URL}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --corpus-locations --corpus-limit 500 ${DB_FLAG}
' || echo "corpus_locations: previous run still active, skipping"
EOF
chmod +x /app/corpus_locations.sh

# Nemesis discovery: add opponents I've lost to
cat > /app/corpus_nemeses.sh << EOF
#!/bin/sh
exec flock -n ${LOCKDIR}/corpus_nemeses.lock sh -c '
cd /app
export CR_PLAYER_TAG="${CR_PLAYER_TAG}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --corpus-nemeses --player-tag "${CR_PLAYER_TAG}" ${DB_FLAG}
' || echo "corpus_nemeses: previous run still active, skipping"
EOF
chmod +x /app/corpus_nemeses.sh

# Combined corpus scrape: battles + replays in one pass
cat > /app/corpus_combined.sh << EOF
#!/bin/sh
exec flock -n ${LOCKDIR}/corpus_combined.lock sh -c '
cd /app
export CR_API_KEY="${CR_API_KEY}"
export CR_PLAYER_TAG="${CR_PLAYER_TAG}"
[ -n "${CR_API_URL}" ] && export CR_API_URL="${CR_API_URL}"
export BROWSER_WS_URL="${BROWSER_WS_URL:-http://cr-browser:9223}"
export ROYALEAPI_SESSION_PATH="${ROYALEAPI_SESSION_PATH:-/app/data/royaleapi_session.json}"
export REPLAYS_PER_PLAYER="${REPLAYS_PER_PLAYER:-25}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
${SCRAPER_ENV_EXPORTS}
export PYTHONUNBUFFERED=1
clash-stats --corpus-combined --corpus-limit 50 --concurrency 12 --max-pages 3 ${DB_FLAG}
' || echo "corpus_combined: previous run still active, skipping"
EOF
chmod +x /app/corpus_combined.sh

# Corpus replays — SLOW trickle (decoupled from corpus_scrape, which pulls
# battles via the official CR API and is unaffected by RoyaleAPI/Cloudflare).
# Corpus-scale replay volume (50 players × concurrency 12 every minute) burned
# exits faster than the pool recovered. This is the opposite end of the dial: a
# few players' freshest replays, fully gentle (1 req/s, low concurrency, first
# battle page only), every 5 min. Bumped 3→8 players / 10→5 min on 06-10 after
# a clean hour (42 replays, 0 challenges/failures) showed the exit ~95% idle;
# headroom remains, still far below the corpus-scale config that thrashed. Some
# corpus replay data gathered slowly —
# the counterfactual sim needs OTHER players' games — beats none. A challenged
# exit triggers a cooldown-guarded reactive rotation between passes.
#
# Uses --corpus-combined (the HTTP replay path via fetch_replays_http, routed
# through the VPN proxy) — NOT --corpus-replays, which is the legacy Playwright/
# cr-browser path. CR_API_KEY is baked (the combined pass also refreshes the 3
# players' battles via the official API). CR_PLAYER_TAG is deliberately NOT
# baked so personal_tag stays None — personal replays are owned by
# personal_combined; this job is corpus-only.
cat > /app/corpus_replays.sh << EOF
#!/bin/sh
exec flock -n ${LOCKDIR}/corpus_replays.lock sh -c '
cd /app
export CR_API_KEY="${CR_API_KEY}"
[ -n "${CR_API_URL}" ] && export CR_API_URL="${CR_API_URL}"
export ROYALEAPI_SESSION_PATH="${ROYALEAPI_SESSION_PATH:-/app/data/royaleapi_session.json}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
${SCRAPER_ENV_EXPORTS}
export ROYALEAPI_REQUESTS_PER_SEC="${CORPUS_REPLAY_RATE:-1.0}"
export PYTHONUNBUFFERED=1
clash-stats --corpus-combined --corpus-limit 8 --replays-per-player 8 --max-pages 1 --concurrency 2 ${DB_FLAG}
' || echo "corpus_replays: previous run still active, skipping"
EOF
chmod +x /app/corpus_replays.sh

# Incremental WP inference: process games with replays but no WP data
cat > /app/wp_infer_new.sh << EOF
#!/bin/sh
exec flock -n ${LOCKDIR}/wp_infer_new.lock sh -c '
cd /app
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --wp-infer-new ${DB_FLAG}
' || echo "wp_infer_new: previous run still active, skipping"
EOF
chmod +x /app/wp_infer_new.sh

# Manual/on-demand VPN exit rotation (clash-stats --rotate-exit, force=True).
# No longer cron-scheduled: corpus is battles-only, so there's no sustained
# replay volume to spread, and periodic rotation needlessly invalidates the
# personal path's IP-bound cf_clearance. Kept as a tool for manually rolling
# the exit. Holds both scrape locks during the ~10s reconnect so no fetch runs
# against a half-rotated tunnel.
cat > /app/rotate_exit.sh << EOF
#!/bin/sh
exec flock -n ${LOCKDIR}/corpus_combined.lock flock -n ${LOCKDIR}/personal_combined.lock sh -c '
cd /app
${SCRAPER_ENV_EXPORTS}
export PYTHONUNBUFFERED=1
clash-stats --rotate-exit ${DB_FLAG}
' || echo "rotate_exit: scrape in progress, skipping"
EOF
chmod +x /app/rotate_exit.sh

# Incremental TCN embedding (new games only, no retraining)
cat > /app/embed_new.sh << EOF
#!/bin/sh
exec flock -n ${LOCKDIR}/embed_new.lock sh -c '
cd /app
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --embed-new ${DB_FLAG}
' || echo "embed_new: previous run still active, skipping"
EOF
chmod +x /app/embed_new.sh

# TCN retraining
cat > /app/tcn_train.sh << EOF
#!/bin/sh
exec flock -n ${LOCKDIR}/tcn_train.lock sh -c '
cd /app
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --train-tcn ${DB_FLAG}
' || echo "tcn_train: previous run still active, skipping"
EOF
chmod +x /app/tcn_train.sh

# Activity model retraining
cat > /app/train_activity.sh << EOF
#!/bin/sh
exec flock -n ${LOCKDIR}/train_activity.lock sh -c '
cd /app
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --train-activity-model ${DB_FLAG}
' || echo "train_activity: previous run still active, skipping"
EOF
chmod +x /app/train_activity.sh

echo "=== cr-tracker starting ==="
echo "  Player tag: #${CR_PLAYER_TAG}"
echo "  API:        ${CR_API_URL:-https://api.clashroyale.com/v1}"
echo "  Personal:   every 2 min combined (battles + replays, atomic)"
echo "  Database:   ${DATABASE_URL}"
echo "  Dashboard:  http://0.0.0.0:8078"
echo "  Corpus:     every 1 min combined (battles + replays, 50 players, 12 tabs)"
echo "  Discovery:  daily 3am opponent network + weekly Mon 7am regional leaderboards"
echo "  Metrics:    http://0.0.0.0:8001/metrics (Prometheus)"
echo "  noVNC:      http://0.0.0.0:6080 (browser sidecar)"

# Initial fetch on startup
/app/fetch.sh

# Start dashboard via gunicorn (threaded for concurrent requests)
export CR_DB_PATH="${DATABASE_URL}"
gunicorn "tracker.dashboard:create_app()" \
    --bind 0.0.0.0:8078 \
    --workers 2 \
    --threads 4 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - &
FLASK_PID=$!
trap "kill ${FLASK_PID} 2>/dev/null; exit 0" TERM INT

# Start cron in foreground
echo "=== cron active ==="
cron -f
