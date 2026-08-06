"""Add the v10 deck prior to an existing WP shard set, in place.

The deck arrays are new information; card_ids, features, lengths and labels are
untouched by the v10 change. Rebuilding all 1.68M rows to obtain two small
per-game arrays would burn hours recomputing features that are already correct,
and — worse — a fresh build selects a fresh battle set, because the corpus grows
continuously. Backfilling in battle_ids.txt order keeps the v10 shard set
row-identical to the one v9 trained on, so the A/B differs only by the feature
under test.

Safe to run against the live shard dir: the reader treats deck_ids.npy as
optional and a v9-shaped model (deck_features=False) ignores the tensors even
when they are present.

Run with cwd=/app:
  PYTHONPATH=/app/src python3 tools/ml/backfill_shard_decks.py [shard_dir]
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sqlalchemy import text

sys.path.insert(0, "/app/src")
from tracker.database import get_engine, get_session          # noqa: E402
from tracker.ml.card_metadata import CardVocabulary           # noqa: E402
from tracker.ml.sequence_dataset import DECK_SIZE, VARIANT_IDX  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tracker.ml.deck_backfill")

CHUNK = 2000


def main() -> None:
    shard_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/wp_shards")
    meta = json.loads((shard_dir / "meta.json").read_text())
    bids = (shard_dir / "battle_ids.txt").read_text().splitlines()
    n = meta["n"]
    if len(bids) != n:
        raise SystemExit(f"battle_ids.txt has {len(bids)} rows but meta says n={n}")

    session = get_session(get_engine(os.environ["DATABASE_URL"]))
    vocab = CardVocabulary(session)
    logger.info("Backfilling deck prior for %d rows in %s (vocab %d)", n, shard_dir, vocab.size)

    ids_mm = np.lib.format.open_memmap(
        shard_dir / "deck_ids.npy.tmp", mode="w+", dtype=np.int16, shape=(n, 2, DECK_SIZE))
    var_mm = np.lib.format.open_memmap(
        shard_dir / "deck_vars.npy.tmp", mode="w+", dtype=np.int8, shape=(n, 2, DECK_SIZE))

    row_of: dict[str, list[int]] = {}
    for i, b in enumerate(bids):
        row_of.setdefault(b, []).append(i)

    missing = 0
    for start in range(0, n, CHUNK):
        chunk = bids[start:start + CHUNK]
        rows = session.execute(
            text("""
                SELECT battle_id, is_player_deck, card_name, card_variant
                FROM deck_cards WHERE battle_id IN :bids
                ORDER BY battle_id, is_player_deck DESC, card_name
            """),
            {"bids": tuple(set(chunk))},
        ).all()
        fill: dict[str, list[int]] = {}
        seen = set()
        for bid, is_own, name, variant in rows:
            targets = row_of.get(bid)
            if not targets:
                continue
            seen.add(bid)
            f = fill.setdefault(bid, [0, 0])
            r = 0 if is_own == 1 else 1
            if f[r] >= DECK_SIZE:
                continue          # duplicate deck_cards rows exist for some battles
            for t in targets:
                ids_mm[t, r, f[r]] = vocab.encode(name)
                var_mm[t, r, f[r]] = VARIANT_IDX.get(variant or "base", 0)
            f[r] += 1
        missing += len(set(chunk)) - len(seen)
        if (start // CHUNK) % 50 == 0:
            logger.info("  %d/%d rows (%.1f%%), %d battles with no deck rows",
                        start, n, 100.0 * start / n, missing)

    ids_mm.flush(); var_mm.flush()
    del ids_mm, var_mm
    # Atomic-ish publish: readers key on deck_ids.npy existing, so put the
    # variants in place first.
    os.replace(shard_dir / "deck_vars.npy.tmp", shard_dir / "deck_vars.npy")
    os.replace(shard_dir / "deck_ids.npy.tmp", shard_dir / "deck_ids.npy")

    ids = np.load(shard_dir / "deck_ids.npy", mmap_mode="r")
    nonzero = int((ids[:, 0, :] != 0).sum(axis=1).mean() * 100) / 100
    empty = int((ids.reshape(n, -1) == 0).all(axis=1).sum())
    logger.info("Done: mean %.2f own-deck cards populated, %d rows fully empty, "
                "%d battles had no deck rows", nonzero, empty, missing)

    meta["deck_features"] = True
    meta["deck_size"] = DECK_SIZE
    meta["deck_backfilled"] = True
    (shard_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  deck prior written: {shard_dir}/deck_ids.npy, deck_vars.npy")


if __name__ == "__main__":
    main()
