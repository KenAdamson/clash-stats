"""Phase 0 prerequisite: map every deck_hash in the pilot-embed shards to its
8-card NAME set, so the benchmark can require a real deck distance for
positive pairs instead of trusting hash inequality (hashes differ on evo
levels — two "different decks" can share all 8 cards).

Reads the shard metadata for the distinct hashes, resolves each via ONE
representative battle's is_player_deck=1 rows, writes
data/pilot_embed/deck_cardsets.json  {deck_hash: [8 card names]}.

Read-only; run against the replica to keep the primary clear:
  DATABASE_URL=postgresql://...@192.168.7.62/clash_stats \
  PYTHONPATH=/app/src python3 tools/eval/deck_cardsets_extract.py
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tracker.ml.deck_cardsets")

SHARD_DIR = Path("data/pilot_embed/wp_v9")
OUT = Path("data/pilot_embed/deck_cardsets.json")
BATCH = 500


def main() -> int:
    # The shards already pair every deck_hash with battle_ids — pick ONE
    # representative battle per hash from shard metadata instead of searching
    # `battles` (the original DISTINCT ON scan took ~60s/batch and lost to
    # WAL-replay cancellation on the hot standby). deck_cards.battle_id is
    # indexed, so this is a millisecond lookup per representative.
    rep: dict[str, str] = {}
    for shard in sorted(SHARD_DIR.glob("shard_*.npz")):
        z = np.load(shard, allow_pickle=False)
        for dh, bid in zip(z["deck_hashes"].tolist(), z["battle_ids"].tolist()):
            if dh and dh != "None" and dh not in rep:
                rep[dh] = bid
    logger.info("distinct deck hashes in shards: %d", len(rep))

    done: dict[str, list[str]] = {}
    if OUT.exists():
        done = json.loads(OUT.read_text())
        logger.info("resume: %d already mapped", len(done))
    todo = sorted(set(rep) - set(done))

    bid_to_hash = {rep[dh]: dh for dh in todo}
    bids = sorted(bid_to_hash)
    engine = create_engine(os.environ["DATABASE_URL"])
    with engine.connect() as conn:
        for i in range(0, len(bids), BATCH):
            batch = bids[i:i + BATCH]
            rows = conn.execute(text("""
                SELECT battle_id, card_name FROM deck_cards
                WHERE battle_id = ANY(:bids) AND is_player_deck = 1
            """), {"bids": batch}).all()
            acc: dict[str, set] = {}
            for bid, name in rows:
                acc.setdefault(bid_to_hash[bid], set()).add(name)
            for dh, names in acc.items():
                done[dh] = sorted(names)
            OUT.write_text(json.dumps(done))
            logger.info("mapped %d/%d", len(done), len(rep))
    OUT.write_text(json.dumps(done))

    sizes = [len(v) for v in done.values()]
    n8 = sum(1 for s in sizes if s == 8)
    logger.info("done: %d hashes, %d with exactly 8 cards (%.1f%%)",
                len(done), n8, 100 * n8 / max(len(done), 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
