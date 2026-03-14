#!/bin/sh
set -e

DB_PATH=/app/data/clash_royale_history.db

# SSH setup for git push to GitHub
mkdir -p /root/.ssh
chmod 700 /root/.ssh
# Key is mounted read-only — copy so we can set permissions
if [ -f /root/.ssh/id_ed25519 ]; then
    cp /root/.ssh/id_ed25519 /root/.ssh/id_ed25519_tmp
    mv /root/.ssh/id_ed25519_tmp /root/.ssh/deploy_key
    chmod 600 /root/.ssh/deploy_key
    cat > /root/.ssh/config << SSHEOF
Host github.com
    IdentityFile /root/.ssh/deploy_key
    StrictHostKeyChecking accept-new
SSHEOF
fi
ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null || true
git config --global user.email "cr-tracker@workhorse"
git config --global user.name "cr-tracker"
# .git is mounted at /app/.git — tell git it's safe
git config --global --add safe.directory /app

# Build fetch script with baked-in env vars
# (BusyBox crond runs jobs in a clean environment)
cat > /app/fetch.sh << EOF
#!/bin/sh
export CR_API_KEY="${CR_API_KEY}"
export CR_PLAYER_TAG="${CR_PLAYER_TAG}"
[ -n "${CR_API_URL}" ] && export CR_API_URL="${CR_API_URL}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --fetch --db ${DB_PATH}
EOF
chmod +x /app/fetch.sh

# Build publish wrapper with baked-in env vars for crond
cat > /app/publish_wrapper.sh << EOF
#!/bin/sh
export STATS_REPO_URL="${STATS_REPO_URL}"
export STATS_BRANCH="${STATS_BRANCH:-stats}"
export STATS_REMOTE="${STATS_REMOTE:-origin}"
/app/publish_stats.sh
EOF
chmod +x /app/publish_wrapper.sh

# Build personal combined wrapper: fetch battles only (replays handled by corpus_combined)
cat > /app/personal_combined.sh << EOF
#!/bin/sh
LOCKFILE=/tmp/personal_combined.lock
STALE_MINUTES=10
if [ -f "\$LOCKFILE" ]; then
    lock_age=\$(( \$(date +%s) - \$(date -r "\$LOCKFILE" +%s) ))
    if [ "\$lock_age" -gt \$(( STALE_MINUTES * 60 )) ]; then
        echo "personal_combined: stale lock (\${lock_age}s old), removing"
        rm -f "\$LOCKFILE"
    else
        echo "personal_combined: previous run still active, skipping"
        exit 0
    fi
fi
trap 'rm -f "\$LOCKFILE"' EXIT
touch "\$LOCKFILE"
export CR_API_KEY="${CR_API_KEY}"
export CR_PLAYER_TAG="${CR_PLAYER_TAG}"
[ -n "${CR_API_URL}" ] && export CR_API_URL="${CR_API_URL}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --personal-combined --player-tag "${CR_PLAYER_TAG}" --db ${DB_PATH}
EOF
chmod +x /app/personal_combined.sh

# Build corpus wrapper scripts for crond
cat > /app/corpus_update.sh << EOF
#!/bin/sh
export CR_API_KEY="${CR_API_KEY}"
[ -n "${CR_API_URL}" ] && export CR_API_URL="${CR_API_URL}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --corpus-update --corpus-limit 500 --db ${DB_PATH}
EOF
chmod +x /app/corpus_update.sh

cat > /app/corpus_scrape.sh << EOF
#!/bin/sh
LOCKFILE=/tmp/corpus_scrape.lock
STALE_MINUTES=30
if [ -f "\$LOCKFILE" ]; then
    lock_age=\$(( \$(date +%s) - \$(date -r "\$LOCKFILE" +%s) ))
    if [ "\$lock_age" -gt \$(( STALE_MINUTES * 60 )) ]; then
        echo "corpus_scrape: stale lock (\${lock_age}s old), removing"
        rm -f "\$LOCKFILE"
    else
        echo "corpus_scrape: previous run still active, skipping"
        exit 0
    fi
fi
trap 'rm -f "\$LOCKFILE"' EXIT
touch "\$LOCKFILE"
export CR_API_KEY="${CR_API_KEY}"
[ -n "${CR_API_URL}" ] && export CR_API_URL="${CR_API_URL}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --corpus-scrape --corpus-limit 500 --db ${DB_PATH}
EOF
chmod +x /app/corpus_scrape.sh

cat > /app/sim_refresh.sh << EOF
#!/bin/sh
export CR_PLAYER_TAG="${CR_PLAYER_TAG}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --sim-full --player-tag "${CR_PLAYER_TAG}" --db ${DB_PATH}
EOF
chmod +x /app/sim_refresh.sh

cat > /app/corpus_replays.sh << EOF
#!/bin/sh
LOCKFILE=/tmp/corpus_replays.lock
STALE_MINUTES=30
if [ -f "\$LOCKFILE" ]; then
    lock_age=\$(( \$(date +%s) - \$(date -r "\$LOCKFILE" +%s) ))
    if [ "\$lock_age" -gt \$(( STALE_MINUTES * 60 )) ]; then
        echo "corpus_replays: stale lock (\${lock_age}s old), removing"
        rm -f "\$LOCKFILE"
    else
        echo "corpus_replays: previous run still active, skipping"
        exit 0
    fi
fi
trap 'rm -f "\$LOCKFILE"' EXIT
touch "\$LOCKFILE"
export BROWSER_WS_URL="${BROWSER_WS_URL:-http://cr-browser:9223}"
export ROYALEAPI_SESSION_PATH="${ROYALEAPI_SESSION_PATH:-/app/data/royaleapi_session.json}"
export REPLAYS_PER_PLAYER="${REPLAYS_PER_PLAYER:-25}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --corpus-replays --corpus-limit 500 --concurrency 12 --max-pages 2 --db ${DB_PATH}
EOF
chmod +x /app/corpus_replays.sh

# Network discovery: mine opponent tags and add to corpus
cat > /app/corpus_discover.sh << EOF
#!/bin/sh
export CR_API_KEY="${CR_API_KEY}"
[ -n "${CR_API_URL}" ] && export CR_API_URL="${CR_API_URL}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --corpus-discover --corpus-limit 500 --db ${DB_PATH}
EOF
chmod +x /app/corpus_discover.sh

# Location leaderboard discovery
cat > /app/corpus_locations.sh << EOF
#!/bin/sh
export CR_API_KEY="${CR_API_KEY}"
[ -n "${CR_API_URL}" ] && export CR_API_URL="${CR_API_URL}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --corpus-locations --corpus-limit 500 --db ${DB_PATH}
EOF
chmod +x /app/corpus_locations.sh

# Nemesis discovery: add opponents I've lost to
cat > /app/corpus_nemeses.sh << EOF
#!/bin/sh
export CR_PLAYER_TAG="${CR_PLAYER_TAG}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --corpus-nemeses --player-tag "${CR_PLAYER_TAG}" --db ${DB_PATH}
EOF
chmod +x /app/corpus_nemeses.sh

# Combined corpus scrape: battles + replays in one pass
cat > /app/corpus_combined.sh << EOF
#!/bin/sh
LOCKFILE=/tmp/corpus_combined.lock
STALE_MINUTES=30
if [ -f "\$LOCKFILE" ]; then
    lock_age=\$(( \$(date +%s) - \$(date -r "\$LOCKFILE" +%s) ))
    if [ "\$lock_age" -gt \$(( STALE_MINUTES * 60 )) ]; then
        echo "corpus_combined: stale lock (\${lock_age}s old), removing"
        rm -f "\$LOCKFILE"
    else
        echo "corpus_combined: previous run still active, skipping"
        exit 0
    fi
fi
trap 'rm -f "\$LOCKFILE"' EXIT
touch "\$LOCKFILE"
export CR_API_KEY="${CR_API_KEY}"
export CR_PLAYER_TAG="${CR_PLAYER_TAG}"
[ -n "${CR_API_URL}" ] && export CR_API_URL="${CR_API_URL}"
export BROWSER_WS_URL="${BROWSER_WS_URL:-http://cr-browser:9223}"
export ROYALEAPI_SESSION_PATH="${ROYALEAPI_SESSION_PATH:-/app/data/royaleapi_session.json}"
export REPLAYS_PER_PLAYER="${REPLAYS_PER_PLAYER:-25}"
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --corpus-combined --corpus-limit 500 --concurrency 12 --max-pages 2 --db ${DB_PATH}
EOF
chmod +x /app/corpus_combined.sh

# TCN retraining
cat > /app/tcn_train.sh << EOF
#!/bin/sh
[ -n "${DATABASE_URL}" ] && export DATABASE_URL="${DATABASE_URL}"
export PYTHONUNBUFFERED=1
clash-stats --train-tcn --db ${DB_PATH}
EOF
chmod +x /app/tcn_train.sh

PUSH_DEST="${STATS_REPO_URL:-origin}/${STATS_BRANCH:-stats}"
echo "=== cr-tracker starting ==="
echo "  Player tag: #${CR_PLAYER_TAG}"
echo "  API:        ${CR_API_URL:-https://api.clashroyale.com/v1}"
echo "  Personal:   every 2 min combined (battles + replays, atomic)"
echo "  Database:   ${DB_PATH}"
echo "  Dashboard:  http://0.0.0.0:8078"
echo "  Stats push: every 5 min → ${PUSH_DEST}"
echo "  Corpus:     every 5 min combined (battles + replays, 500 players, 12 tabs)"
echo "  Discovery:  daily 3am opponent network + weekly Mon 7am regional leaderboards"
echo "  Metrics:    http://0.0.0.0:8001/metrics (Prometheus)"
echo "  noVNC:      http://0.0.0.0:6080 (browser sidecar)"

# Initial fetch on startup
/app/fetch.sh

# Start Flask dashboard in background
export CR_DB_PATH="${DB_PATH}"
python -m tracker.dashboard &
FLASK_PID=$!
trap "kill ${FLASK_PID} 2>/dev/null; exit 0" TERM INT

# Start cron in foreground
echo "=== cron active ==="
cron -f
