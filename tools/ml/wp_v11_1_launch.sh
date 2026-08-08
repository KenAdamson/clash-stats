#!/bin/bash
# WP v10 training run: the deck prior.
#
# Warm-starts from v9, whose encoder is exactly what we want to keep — v10
# changes only what the head sees (both decks, mean-pooled card+variant
# embeddings, broadcast to every tick). WP_RESUME does a shape-compatible
# transfer, so the card embedding and all six TCN blocks carry over and only the
# grown head plus the new variant table initialise fresh. That is why this run
# should finish in well under v9's 46 hours.
#
# Guarded by xpu_train.lock: the A770 is small-BAR and cannot hold two training
# jobs — a Sunday tcn_train cron colliding with the variant-B run killed both 13
# seconds apart. The GPU crons are paused in the repo crontab (#V10BUILD#) for
# the duration, but the lock is the belt to that braces.
#
# Batch 512, which is what v9 (the 2x-width variant-B run, 3.83M params) itself
# used. The "batch 1024 is the proven sustained max" result was measured on the
# ~974K-param model and does NOT carry to this width: at 1024 a 3.83M-param v10
# dies with UR_RESULT_ERROR_OUT_OF_HOST_MEMORY in the backward pass, because the
# 256MB small-BAR host-visible window caps TOTAL host-backed memory once Adam
# moments and the gradient graph accumulate. Relaxed-alloc lifts only the 4GB
# SINGLE-allocation ceiling, not this.
#
# Usage:  nohup bash /app/tools/ml/wp_v10_launch.sh &
set -u

log=/app/data/wp_v11_1.log
LOCKDIR=${LOCKDIR:-/var/lock}
XPU_TRAIN_LOCK=${XPU_TRAIN_LOCK:-${LOCKDIR}/xpu_train.lock}

export DATABASE_URL="${DATABASE_URL:-postgresql://clash_stats:clash_stats_pw_2026@clash-postgres/clash_stats}"
export PYTHONPATH=/app/src PYTHONUNBUFFERED=1
export CR_PLAYER_TAG="${CR_PLAYER_TAG:-L90009GPP}"

# XPU: lift the 4GB single-allocation ceiling so batch 1024 is reachable at all.
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 UR_L0_USE_RELAXED_ALLOCATION_LIMITS=1
export IGC_ExtraOCLOptions="-cl-intel-greater-than-4GB-buffer-required"
export UR_L0_USE_IMMEDIATE_COMMANDLISTS=1

# Architecture must match v9 exactly or the warm start is refused.
export WP_TCN_CHANNELS=128,128,256,256,512,512
export WP_CARD_EMBED=32
export WP_SHARD_DIR=/app/data/wp_shards
export WP_FROM_SCRATCH=1
export WP_DECK_FEATURES=1
export WP_DECK_INTERACTION=0
export WP_DECK_ANTISYM=1
# Reweighting stays OFF. It distorts the training distribution, and per Ken it
# should be forced on us by evidence rather than opened with -- rare is not bad,
# and his own ladder deck is combination-unique but shape-typical (weight 1.257).
export WP_RARITY_WEIGHTS=0
export WP_BATCH_SIZE=512
# Half the from-scratch default at this batch: a warm start restarts Adam and
# the LR schedule, so the full rate would knock the loaded weights out of the
# basin they already found.
export WP_LR=1.5e-4
export WP_RESUME=/app/data/ml_models/wp_v10.pt

cd /app || exit 1

if [ ! -f "${WP_RESUME}" ]; then
  echo "[v11.1 $(date -u +%FT%TZ)] FATAL: warm-start checkpoint ${WP_RESUME} missing" >> "$log"
  exit 1
fi
if [ ! -f /app/data/wp_shards/deck_ids.npy ]; then
  echo "[v11.1 $(date -u +%FT%TZ)] FATAL: shards carry no deck prior — run" \
       "tools/ml/backfill_shard_decks.py first" >> "$log"
  exit 1
fi

{
  echo "[v11.1 $(date -u +%FT%TZ)] starting: deck prior, warm start from $(basename "${WP_RESUME}")"
  echo "[v10] batch=${WP_BATCH_SIZE} lr=${WP_LR} tcn=${WP_TCN_CHANNELS} embed=${WP_CARD_EMBED}"
} >> "$log"

# NOT `exec flock ... || echo`: exec replaces the shell, so the fallback branch
# would never run and a lock-contention skip would vanish silently.
#
# -E 99 gives lock contention its own exit code. Without it flock returns 1 both
# when the lock is held AND when the command itself exits 1, so the first v10
# crash (an XPU OOM inside training) was logged as "another job is live" — a
# misdiagnosis that sent me looking at the lock instead of at the model.
flock -n -E 99 "${XPU_TRAIN_LOCK}" clash-stats --train-wp >> "$log" 2>&1
rc=$?
if [ $rc -eq 99 ]; then
  echo "[v11.1 $(date -u +%FT%TZ)] could not take ${XPU_TRAIN_LOCK} — another XPU training job is live" >> "$log"
elif [ $rc -ne 0 ]; then
  echo "[v11.1 $(date -u +%FT%TZ)] TRAINING FAILED rc=${rc} — see traceback above" >> "$log"
fi
echo "[v11.1 $(date -u +%FT%TZ)] exited rc=${rc}" >> "$log"
exit $rc
