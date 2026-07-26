#!/bin/bash
# CAPACITY EXPERIMENT — variant B: 2x-width TCN (128/128/256/256/512/512) + embed 32
# (~3.83M params, 3.9x wp_v8). Batch 512 / LR 3e-4 defaults (proven to fit small-BAR
# sustained). Full 1.67M shards. Queues BEHIND the data-iso run (XPU is single-tenant).
log=/app/data/wp_capacityB.log
: "${DATABASE_URL:?DATABASE_URL must be set — supplied by docker-compose; docker exec inherits the container env}"
export DATABASE_URL   # ensure child processes inherit it even if it arrived unexported
export PYTHONPATH=/app/src PYTHONUNBUFFERED=1
running(){ for p in /proc/[0-9]*; do case "$(readlink "$p/exe" 2>/dev/null)" in *python*) case "$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)" in *--train-wp*) return 0;; esac;; esac; done; return 1; }
echo "[capB $(date -u +%H:%M:%S)] waiting for current train-wp (data-iso) to finish before starting B" >> "$log"
while running; do sleep 300; done
echo "[capB $(date -u +%H:%M:%S)] XPU free — launching variant B (2x width, embed 32)" >> "$log"
cd /app || exit 1
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 UR_L0_USE_RELAXED_ALLOCATION_LIMITS=1
export IGC_ExtraOCLOptions="-cl-intel-greater-than-4GB-buffer-required" UR_L0_USE_IMMEDIATE_COMMANDLISTS=1
export WP_SHARD_DIR=/app/data/wp_shards WP_FROM_SCRATCH=1 CR_PLAYER_TAG=L90009GPP
export WP_TCN_CHANNELS=128,128,256,256,512,512 WP_CARD_EMBED=32
exec clash-stats --train-wp >> "$log" 2>&1
