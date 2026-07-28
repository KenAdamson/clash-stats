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

# --- Corpus replay egress: DIRECT (2026-07-26) -----------------------------
# The 2026-06 RoyaleAPI ban was pinned to the OLD residential IP. After the
# router cutover the new WAN IP is unbanned: direct requests get an ordinary
# Cloudflare challenge (cf-mitigated: challenge, not block) and, with a
# residential-minted cf_clearance, return clean 200s. Meanwhile the shared
# Mullvad exit is heavily challenged (~600 403s/day, yield ~0 replays/cycle).
# Corpus replays therefore egress DIRECT; personal/alt stay on the VPN, so a
# re-ban would cost corpus throughput (recoverable) rather than personal
# battle history (not recoverable).
#
# cf_clearance is bound to the egress IP, so the direct path MUST use its own
# session file — otherwise the two paths keep invalidating each other's cookie.
# The Discord login cookie (__royaleapi_session_v2) is NOT IP-bound and is
# seeded by copying the VPN session file on first use.
#
# Set CORPUS_REPLAY_DIRECT=0 to fall back to the VPN path.
CORPUS_REPLAY_DIRECT="${CORPUS_REPLAY_DIRECT:-1}"
CORPUS_SESSION_PATH="${ROYALEAPI_SESSION_PATH:-/app/data/royaleapi_session.json}"
CORPUS_SCRAPER_ENV="${SCRAPER_ENV_EXPORTS}"
if [ "${CORPUS_REPLAY_DIRECT}" = "1" ]; then
    CORPUS_SESSION_PATH="${DIRECT_SESSION_PATH:-/app/data/royaleapi_session_direct.json}"
    # Empty proxy/control-url = no VPN, no exit rotation. The standalone
    # `flaresolverr` container egresses residential; cr-scraper-fs is VPN-bound.
    CORPUS_SCRAPER_ENV="export ROYALEAPI_PROXY=\"\"
export FLARESOLVERR_URL=\"${DIRECT_FLARESOLVERR_URL:-http://flaresolverr:8191/v1}\"
export GLUETUN_CONTROL_URL=\"\"
"
    # Seed the direct session from the VPN one so the login cookie carries over;
    # cf_clearance is re-minted against the residential IP on first use.
    if [ ! -f "${CORPUS_SESSION_PATH}" ] && [ -f "${ROYALEAPI_SESSION_PATH:-/app/data/royaleapi_session.json}" ]; then
        cp "${ROYALEAPI_SESSION_PATH:-/app/data/royaleapi_session.json}" "${CORPUS_SESSION_PATH}"
        echo "entrypoint: seeded direct corpus session at ${CORPUS_SESSION_PATH}"
    fi
    echo "entrypoint: corpus replays egress DIRECT (session ${CORPUS_SESSION_PATH})"
else
    echo "entrypoint: corpus replays egress via VPN proxy"
fi

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

# Rising-star sampler: re-run the top-of-band seed WITHOUT --reseed-archive so
# newly-risen players get added to the active corpus while existing tracked
# players stay (dormancy hygiene handles attrition). Keeps the corpus tracking
# the current top-5%-of-band cohort as players climb through the bands.
cat > /app/reseed_risers.sh << EOF
#!/bin/sh
exec flock -n ${LOCKDIR}/reseed_risers.lock sh -c '
cd /app
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --reseed-top-of-band ${DB_FLAG}
' || echo "reseed_risers: previous run still active, skipping"
EOF
chmod +x /app/reseed_risers.sh

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
# bumped 8→16 players + 8→12 replays/player on 07-07 (inventory-driven
# selection restored demand; measured 234/hr, one transient 429 burst).
# Depth experiment (8×25×2pages, 07-08) FAILED at 94/hr: old unfetched
# stock is DEAD — RoyaleAPI only caches replays near battle-time, so
# ranking by deep inventory aims visits at unfetchable backlog. Fix:
# selection ranks by FRESH inventory (<14h) and breadth wins (16×12×1pg).
# Request budget ~208 req/run ≈ 70% of window at 1 req/s — depth vs
# breadth is CF-risk-neutral at fixed budget; risk scales with rate ×
# sustained occupancy. Do NOT push occupancy near 100% on one exit;
# 10x = exit-rotation service, held in reserve per Ken 07-07. A challenged
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
export ROYALEAPI_SESSION_PATH="${CORPUS_SESSION_PATH}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
${CORPUS_SCRAPER_ENV}
export ROYALEAPI_REQUESTS_PER_SEC="${CORPUS_REPLAY_RATE:-1.0}"
export PYTHONUNBUFFERED=1
clash-stats --corpus-combined --corpus-limit ${CORPUS_REPLAY_LIMIT:-16} --replays-per-player 12 --max-pages 1 --concurrency 2 ${DB_FLAG}
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

# --- Shared GPU lock -------------------------------------------------------
# Every XPU job used to hold only its OWN lock, so nothing serialized them
# against each other and the A770 had no lock representing it. On 2026-07-26
# the weekly tcn_train cron fired at 04:00 while a WP capacity run was at
# epoch 25 and BOTH died 13s apart with UR_RESULT_ERROR_OUT_OF_HOST_MEMORY —
# the card's small-BAR host-memory ceiling can't hold two training jobs.
#
# ${XPU_TRAIN_LOCK} is the outer lock every *training* job takes, in addition
# to its own per-job lock (which still prevents self-overlap). Inference is
# deliberately NOT gated: wp_infer_new grabs an XPU context ~18x/hour and
# coexisted with a full training run for 25 epochs without trouble, and
# blocking it behind a multi-day train would stall dashboard freshness.
#
# -n (non-blocking): a weekly retrain that collides simply skips and runs next
# week, which is far cheaper than crashing a multi-day training run.
XPU_TRAIN_LOCK=${LOCKDIR}/xpu_train.lock

# TCN retraining
# NB: no `exec` here — `exec flock ... || echo` makes the fallback unreachable,
# because exec replaces the shell before the `||` can run. These jobs now skip
# routinely (whenever the XPU lock is held), and a weekly retrain skipping
# silently for weeks would be invisible in docker logs.
cat > /app/tcn_train.sh << EOF
#!/bin/sh
flock -n ${XPU_TRAIN_LOCK} flock -n ${LOCKDIR}/tcn_train.lock sh -c '
cd /app
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --train-tcn ${DB_FLAG}
' || echo "tcn_train: XPU busy or previous run still active, skipping"
EOF
chmod +x /app/tcn_train.sh

# WP drift tripwire (weekly): evaluate the PRODUCTION model on the newest
# games and warn if accuracy sags below its own baseline. This is what makes
# "hold off retraining until the corpus doubles" a monitored position instead
# of a bet. Takes ${XPU_TRAIN_LOCK} non-blocking: it is a brief inference
# pass, but on small-BAR the card cannot host it beside a training run — if a
# capacity run is live, skipping a weekly check is the correct trade.
cat > /app/wp_drift_check.sh << EOF
#!/bin/sh
flock -n ${XPU_TRAIN_LOCK} flock -n ${LOCKDIR}/wp_drift_check.lock sh -c '
cd /app
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONPATH=/app/src PYTHONUNBUFFERED=1
python3 tools/eval/wp_drift_check.py
' || echo "wp_drift_check: XPU busy or previous run still active, skipping"
EOF
chmod +x /app/wp_drift_check.sh

# Activity model retraining
# NOT gated on ${XPU_TRAIN_LOCK}: activity_model.py trains a sklearn
# GradientBoostingClassifier — CPU only, no torch, no XPU — so it does not
# compete with WP/TCN training for the card. Gating it would have blocked the
# weekly retrain for the whole duration of any multi-day training run.
cat > /app/train_activity.sh << EOF
#!/bin/sh
flock -n ${LOCKDIR}/train_activity.lock sh -c '
cd /app
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --train-activity-model ${DB_FLAG}
' || echo "train_activity: XPU busy or previous run still active, skipping"
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
