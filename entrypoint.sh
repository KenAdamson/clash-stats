#!/bin/sh
set -e

DB_PATH=/app/data/clash_royale_history.db

# Build fetch script with baked-in env vars
# (BusyBox crond runs jobs in a clean environment)
cat > /app/fetch.sh << EOF
#!/bin/sh
export CR_API_KEY="${CR_API_KEY}"
export CR_PLAYER_TAG="${CR_PLAYER_TAG}"
export PYTHONUNBUFFERED=1
python /app/cr_tracker.py --fetch --db ${DB_PATH}
EOF
chmod +x /app/fetch.sh

echo "=== cr-tracker starting ==="
echo "  Player tag: #${CR_PLAYER_TAG}"
echo "  Schedule:   every 4 hours (crond)"
echo "  Database:   ${DB_PATH}"

# Initial fetch on startup
/app/fetch.sh

# Start BusyBox crond in foreground, log to stderr
echo "=== crond active ==="
crond -f -l 6
