#!/bin/bash
# DATA-ISOLATION run: full 1.67M at the SAME hyperparams as wp_v7 (batch 512,
# LR 3e-4 — the from-scratch defaults, so WP_BATCH_SIZE/WP_LR are UNSET). Removes
# the LR/batch confound so we can attribute any 12k+ change to DATA alone.
# Firehose shards + XPU speed env vars (these affect throughput, NOT the model).
log=/app/data/wp_dataiso.log
: "${DATABASE_URL:?DATABASE_URL must be set — supplied by docker-compose; docker exec inherits the container env}"
export DATABASE_URL   # ensure child processes inherit it even if it arrived unexported
export PYTHONPATH=/app/src PYTHONUNBUFFERED=1
running(){ for p in /proc/[0-9]*; do case "$(readlink "$p/exe" 2>/dev/null)" in *python*) case "$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)" in *--train-wp*) return 0;; esac;; esac; done; return 1; }
if running; then echo "[iso $(date -u +%H:%M:%S)] train-wp already running; abort" >> "$log"; exit 0; fi
echo "[iso $(date -u +%H:%M:%S)] launching data-isolation run (full 1.67M, batch 512, LR 3e-4)" >> "$log"
cd /app || exit 1
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 UR_L0_USE_RELAXED_ALLOCATION_LIMITS=1
export IGC_ExtraOCLOptions="-cl-intel-greater-than-4GB-buffer-required" UR_L0_USE_IMMEDIATE_COMMANDLISTS=1
export WP_SHARD_DIR=/app/data/wp_shards WP_FROM_SCRATCH=1 CR_PLAYER_TAG=L90009GPP
exec clash-stats --train-wp >> "$log" 2>&1
