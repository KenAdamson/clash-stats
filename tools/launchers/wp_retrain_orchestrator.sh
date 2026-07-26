#!/bin/bash
# Wait for the current TCN train (XPU + ~25GB RAM) to drain and for RAM
# headroom, then launch the WP v5 retrain into a cleared field. Written to run
# detached inside cr-tracker so it survives the exec session.
log=/app/data/wp_v5_train.log

# Count real python clash-stats processes matching a pattern, excluding this
# script's own PID (a naive `grep pat /proc/*/cmdline` self-matches its own argv,
# which both false-positives the guard and would hang the drain loop forever).
count_proc() {  # $1 = substring to match in the python cmdline
  local pat="$1" n=0 p pid cmd
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    [ "$pid" = "$$" ] && continue
    cmd=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
    case "$cmd" in
      *python*clash-stats*"$pat"*) n=$((n+1)) ;;
    esac
  done
  echo "$n"
}

if [ "$(count_proc '--train-wp')" -gt 0 ]; then
  echo "[orch $(date -u +%H:%M:%S)] a --train-wp is already running; aborting orchestrator" >> "$log"
  exit 0
fi

echo "[orch $(date -u +%H:%M:%S)] waiting: tcn_train to drain + MemAvailable>=20GB" >> "$log"
while true; do
  tcn=$(count_proc '--train-tcn')
  avail=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
  if [ "$tcn" -eq 0 ] && [ "$avail" -ge 20 ]; then
    echo "[orch $(date -u +%H:%M:%S)] CLEAR: tcn_procs=$tcn availGB=$avail — launching retrain" >> "$log"
    break
  fi
  echo "[orch $(date -u +%H:%M:%S)] waiting: tcn_procs=$tcn availGB=$avail" >> "$log"
  sleep 60
done

cd /app || exit 1
: "${DATABASE_URL:?DATABASE_URL must be set — supplied by docker-compose; docker exec inherits the container env}"
export DATABASE_URL   # ensure child processes inherit it even if it arrived unexported
export CR_PLAYER_TAG="L90009GPP"
export PYTHONUNBUFFERED=1
exec clash-stats --train-wp >> "$log" 2>&1
