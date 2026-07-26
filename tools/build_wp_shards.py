"""Build the WP memmap training shards (see tracker/ml/wp_shard_cache.py).

Usage (in cr-tracker):
  PYTHONPATH=/app/src DATABASE_URL=... python3 build_wp_shards.py [out_dir] [max_games]
Defaults: out_dir=/app/data/wp_shards, max_games=all.
"""
import os
import sys
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from tracker.ml.wp_shard_cache import build_shards

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

out_dir = sys.argv[1] if len(sys.argv) > 1 else "/app/data/wp_shards"
max_games = int(sys.argv[2]) if len(sys.argv) > 2 else None

engine = create_engine(os.environ["DATABASE_URL"])
with Session(engine) as session:
    meta = build_shards(session, out_dir, extra_features=True, max_games=max_games)
print(f"RESULT: {meta}")
