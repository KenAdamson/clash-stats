#!/bin/sh
set -e

DB_PATH=/app/data/clash_royale_history.db

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

echo "=== cr-tracker starting ==="
echo "  Player tag: #${CR_PLAYER_TAG}"
echo "  API:        ${CR_API_URL:-https://api.clashroyale.com/v1}"
echo "  Schedule:   every 4 hours (crond)"
echo "  Database:   ${DB_PATH}"
echo "  Dashboard:  http://0.0.0.0:8078"

# Initial fetch on startup
/app/fetch.sh

# Start Flask dashboard in background
export CR_DB_PATH="${DB_PATH}"
python -m tracker.dashboard &
FLASK_PID=$!
trap "kill ${FLASK_PID} 2>/dev/null; exit 0" TERM INT

# Start BusyBox crond in foreground, log to stderr
echo "=== crond active ==="
crond -f -l 6
