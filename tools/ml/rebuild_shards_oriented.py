"""Rebuild WP shards with the arena-orientation fix, on the SAME battle set.

The point of this build is to isolate one change: ~50% of replays arrive
180-degrees rotated, which randomised arena_x, arena_y and the lane one-hot
across half the corpus. Re-selecting battles would pull in everything ingested
since the original build and confound the comparison, so the exact battle set
and ORDER are replayed from the existing shard's battle_ids.txt.

Everything else -- feature pipeline, max_len, extra_features -- is unchanged,
so a diff between the two shard sets should touch only features 6, 7 and 8:10.

Run with cwd=/app:
  PYTHONPATH=/app/src python3 tools/ml/rebuild_shards_oriented.py <out_dir>
"""
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, "/app/src")
from sqlalchemy import text                                    # noqa: E402
from tracker.database import get_engine, get_session           # noqa: E402
from tracker.ml.card_metadata import CardVocabulary            # noqa: E402
from tracker.ml.wp_shard_cache import build_shards             # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
SRC = Path("data/wp_shards")


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "data/wp_shards_oriented"
    bids = (SRC / "battle_ids.txt").read_text().splitlines()
    meta = json.loads((SRC / "meta.json").read_text())
    print("source shards: %d rows, L=%d, F=%d" % (meta["n"], meta["max_len"], meta["feature_dim"]))

    session = get_session(get_engine(os.environ["DATABASE_URL"]))
    # Results, then re-emit rows in the ORIGINAL order so row i means the same
    # game in both shard sets and the two are directly comparable.
    res = {}
    for i in range(0, len(bids), 5000):
        for bid, r in session.execute(
            text("SELECT battle_id, result FROM battles WHERE battle_id = ANY(:b)"),
            {"b": bids[i:i + 5000]},
        ).all():
            res[bid] = r
    battle_rows = [(b, res[b]) for b in bids if b in res]
    missing = len(bids) - len(battle_rows)
    print("replaying %d battles in original order (%d no longer resolvable)"
          % (len(battle_rows), missing))
    if missing > len(bids) * 0.01:
        raise SystemExit("too many battles missing — the comparison would not be clean")

    vocab = CardVocabulary(session)
    m = build_shards(session, out, vocab=vocab, extra_features=True,
                     battle_rows=battle_rows, max_len=meta["max_len"])
    print("built:", json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
