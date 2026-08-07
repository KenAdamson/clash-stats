#!/bin/bash
# Deploy tracker source into BOTH python trees inside cr-tracker.
#
# The CLI runs the INSTALLED package at site-packages; PYTHONPATH=/app/src runs
# the other one. Copying to only one silently runs stale code, and the failure
# is invisible -- a missing env-var branch just means a feature is quietly off.
# This has bitten three times in one session: migration 012 (37h ingest outage),
# win_probability.py, and wp_training.py (a training run launched WITHOUT the
# interaction term it was supposed to be testing).
#
# Usage: tools/deploy_tracker.sh [file ...]      (default: all of src/tracker)
set -eu
C=${CONTAINER:-cr-tracker}
SP=/usr/local/lib/python3.11/site-packages/tracker
files=("$@")
if [ ${#files[@]} -eq 0 ]; then
  mapfile -t files < <(cd src && find tracker -name '*.py')
  files=("${files[@]/#/src/}")
fi
for f in "${files[@]}"; do
  rel=${f#src/tracker/}
  docker cp "$f" "$C:/app/src/tracker/$rel"
  docker cp "$f" "$C:$SP/$rel"
  echo "  deployed $rel -> both trees"
done
docker exec "$C" bash -lc "find /app/src/tracker $SP -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; true"
echo "  pycache cleared"

# ---------------------------------------------------------------------------
# Crontab deployment is NOT the same as source deployment. Kept here because
# getting it wrong is silent and expensive.
#
# `docker cp crontab cr-tracker:/etc/cron.d/cr-tracker` preserves the HOST
# file's ownership (uid 1000), and Debian cron REFUSES to execute anything in
# /etc/cron.d not owned by root -- with no error, no log line, nothing. It just
# stops running every job in the file. That cost 21.5 hours of total ingest on
# 2026-08-06: the tracker looked healthy, cron was running, the crontab was
# present and correct, and the jobs simply never fired.
#
# Always:
#   docker cp crontab $C:/etc/cron.d/cr-tracker
#   docker exec $C chown root:root /etc/cron.d/cr-tracker
#   docker exec $C chmod 0644     /etc/cron.d/cr-tracker
# then VERIFY by watching for job output, not by reading the file back.
deploy_crontab() {
  docker cp crontab "$C:/etc/cron.d/cr-tracker"
  docker exec "$C" chown root:root /etc/cron.d/cr-tracker
  docker exec "$C" chmod 0644 /etc/cron.d/cr-tracker
  echo "  crontab deployed (root-owned; verify by watching for job output)"
}
