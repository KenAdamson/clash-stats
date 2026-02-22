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

# Build replay wrapper with baked-in env vars for crond
cat > /app/replay_wrapper.sh << EOF
#!/bin/sh
export CR_PLAYER_TAG="${CR_PLAYER_TAG}"
export BROWSER_WS_URL="${BROWSER_WS_URL:-http://cr-browser:9223}"
export ROYALEAPI_SESSION_PATH="${ROYALEAPI_SESSION_PATH:-/app/data/royaleapi_session.json}"
export PYTHONUNBUFFERED=1
clash-stats --fetch-replays --player-tag "${CR_PLAYER_TAG}" --db ${DB_PATH}
EOF
chmod +x /app/replay_wrapper.sh

PUSH_DEST="${STATS_REPO_URL:-origin}/${STATS_BRANCH:-stats}"
echo "=== cr-tracker starting ==="
echo "  Player tag: #${CR_PLAYER_TAG}"
echo "  API:        ${CR_API_URL:-https://api.clashroyale.com/v1}"
echo "  Schedule:   every minute (crond)"
echo "  Database:   ${DB_PATH}"
echo "  Dashboard:  http://0.0.0.0:8078"
echo "  Stats push: every 5 min → ${PUSH_DEST}"
echo "  Replays:    every 30 min (if session active)"
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
