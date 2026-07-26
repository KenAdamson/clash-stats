#!/bin/bash
# Option B: kill the collate-bound in-memory run, build the full-pool memmap
# shards (one-time, ~80 min), then retrain from the firehose (WP_SHARD_DIR).
log=/app/data/wp_firehose.log
: "${DATABASE_URL:?DATABASE_URL must be set — supplied by docker-compose; docker exec inherits the container env}"
export DATABASE_URL   # ensure child processes inherit it even if it arrived unexported
export PYTHONPATH=/app/src PYTHONUNBUFFERED=1

echo "[fh $(date -u +%H:%M:%S)] killing in-memory train-wp (collate-bound)" >> "$log"
for p in /proc/[0-9]*; do
  case "$(readlink "$p/exe" 2>/dev/null)" in *python*)
    case "$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)" in *--train-wp*)
      kill -9 "${p#/proc/}" 2>/dev/null && echo "[fh] killed ${p#/proc/}" >> "$log";; esac;; esac
done
sleep 3

cd /app || exit 1
if [ -f /app/data/wp_shards/meta.json ]; then
  echo "[fh $(date -u +%H:%M:%S)] shards already built — reusing /app/data/wp_shards" >> "$log"
else
  echo "[fh $(date -u +%H:%M:%S)] building full-pool shards -> /app/data/wp_shards" >> "$log"
  rm -rf /app/data/wp_shards
  python3 /app/data/build_wp_shards.py /app/data/wp_shards >> "$log" 2>&1 || {
    echo "[fh $(date -u +%H:%M:%S)] SHARD BUILD FAILED" >> "$log"; exit 1; }
fi

echo "[fh $(date -u +%H:%M:%S)] launching firehose retrain" >> "$log"
# XPU throughput: lift 4GB single-alloc ceiling -> big batches amortize dispatch (~3.5x @2048)
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 UR_L0_USE_RELAXED_ALLOCATION_LIMITS=1
export IGC_ExtraOCLOptions="-cl-intel-greater-than-4GB-buffer-required"
export UR_L0_USE_IMMEDIATE_COMMANDLISTS=1
export WP_SHARD_DIR=/app/data/wp_shards WP_FROM_SCRATCH=1 WP_BATCH_SIZE=1024 WP_LR=6e-4 CR_PLAYER_TAG=L90009GPP
exec clash-stats --train-wp >> "$log" 2>&1
