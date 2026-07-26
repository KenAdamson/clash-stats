#!/bin/bash
# Variant B relaunch — WARM START from the epoch-25 checkpoint that survived the
# 2026-07-26 04:56 crash (val_loss 0.5695 / acc 0.811, already beating wp_v8's
# 0.5732 / 0.805 with no train/val gap, i.e. still improving).
#
# The crash was NOT a capacity limit: the weekly tcn_train cron (0 4 * * 0)
# claimed the XPU at 04:55 and both jobs OOM'd 13s apart. This run therefore
# HOLDS /tmp/locks/xpu_train.lock for its whole life, which tcn_train and
# train_activity now also require -- they will skip rather than evict it.
#
# Optimizer/LR restart on a warm start, so LR is halved (3e-4 -> 1.5e-4) to
# avoid knocking the loaded weights out of the basin they already found.
log=/app/data/wp_capacityB_resume.log
: "${DATABASE_URL:?DATABASE_URL must be set — supplied by docker-compose; docker exec inherits the container env}"
export DATABASE_URL   # ensure child processes inherit it even if it arrived unexported
export PYTHONPATH=/app/src PYTHONUNBUFFERED=1 CR_PLAYER_TAG=L90009GPP
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 UR_L0_USE_RELAXED_ALLOCATION_LIMITS=1
export IGC_ExtraOCLOptions="-cl-intel-greater-than-4GB-buffer-required" UR_L0_USE_IMMEDIATE_COMMANDLISTS=1
export WP_SHARD_DIR=/app/data/wp_shards WP_FROM_SCRATCH=1
export WP_TCN_CHANNELS=128,128,256,256,512,512 WP_CARD_EMBED=32
export WP_RESUME=/app/data/ml_models/wp_capacityB_ep25.pt
export WP_BATCH_SIZE=512 WP_LR=1.5e-4
cd /app || exit 1
echo "[capB-resume $(date -u +%H:%M:%S)] acquiring xpu_train.lock" >> "$log"
exec flock /tmp/locks/xpu_train.lock clash-stats --train-wp >> "$log" 2>&1
