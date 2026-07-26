#!/bin/bash
# Launch the head-injection WP retrain (wp_v6) once there's RAM headroom for the
# ~12GB dataset build (the daily sim_refresh holds ~10GB; XPU is free). Light
# job — frozen encoder, only the head trains — so no cron pausing needed.
log=/app/data/wp_v6_train.log

running() {
  for p in /proc/[0-9]*; do
    case "$(readlink "$p/exe" 2>/dev/null)" in *python*)
      case "$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)" in *--train-wp*) return 0;; esac;; esac
  done
  return 1
}
if running; then echo "[orch $(date -u +%H:%M:%S)] train-wp already running; abort" >> "$log"; exit 0; fi

echo "[orch $(date -u +%H:%M:%S)] waiting for MemAvailable>=15GB" >> "$log"
while :; do
  avail=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
  if [ "$avail" -ge 15 ]; then
    echo "[orch $(date -u +%H:%M:%S)] CLEAR availGB=$avail — launching head-injection retrain" >> "$log"
    break
  fi
  echo "[orch $(date -u +%H:%M:%S)] waiting availGB=$avail" >> "$log"
  sleep 60
done

cd /app || exit 1
: "${DATABASE_URL:?DATABASE_URL must be set — supplied by docker-compose; docker exec inherits the container env}"
export DATABASE_URL   # ensure child processes inherit it even if it arrived unexported
export CR_PLAYER_TAG="L90009GPP"
export PYTHONUNBUFFERED=1
exec clash-stats --train-wp >> "$log" 2>&1
