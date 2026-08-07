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
