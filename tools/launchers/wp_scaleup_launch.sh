#!/bin/bash
# Launch the FULL-POOL WP retrain (PvP + pathOfLegend, all ~1.67M games). The
# streaming dataset build (features per chunk, events freed immediately) makes
# peak memory ~samples-only (~12GB) regardless of pool size, so the gate is 18GB.
# From-scratch 24-dim, NaN-stability fixes on.
log=/app/data/wp_full_train.log
running() { for p in /proc/[0-9]*; do case "$(readlink "$p/exe" 2>/dev/null)" in *python*)
  case "$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)" in *--train-wp*) return 0;; esac;; esac; done; return 1; }
if running; then echo "[orch $(date -u +%H:%M:%S)] train-wp already running; abort" >> "$log"; exit 0; fi
echo "[orch $(date -u +%H:%M:%S)] waiting for MemAvailable>=18GB (full-pool streaming build)" >> "$log"
while :; do
  avail=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
  if [ "$avail" -ge 18 ]; then
    echo "[orch $(date -u +%H:%M:%S)] CLEAR availGB=$avail — launching FULL-POOL retrain" >> "$log"; break
  fi
  echo "[orch $(date -u +%H:%M:%S)] waiting availGB=$avail" >> "$log"; sleep 60
done
cd /app || exit 1
export WP_FROM_SCRATCH=1 CR_PLAYER_TAG=L90009GPP PYTHONUNBUFFERED=1
: "${DATABASE_URL:?DATABASE_URL must be set — supplied by docker-compose; docker exec inherits the container env}"
export DATABASE_URL   # ensure child processes inherit it even if it arrived unexported
exec clash-stats --train-wp >> "$log" 2>&1
