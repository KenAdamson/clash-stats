"""Gate the v10 training run: do the shard deck arrays match the database?

A silently wrong deck array trains a model on garbage for a day before anyone
looks. This samples rows across the shard set and re-derives each deck straight
from deck_cards, comparing card multisets and variants. Any mismatch is fatal.
"""
import os
import random
import sys

import numpy as np
from sqlalchemy import text

sys.path.insert(0, "/app/src")
from tracker.database import get_engine, get_session          # noqa: E402
from tracker.ml.card_metadata import CardVocabulary           # noqa: E402
from tracker.ml.sequence_dataset import DECK_SIZE, VARIANT_IDX  # noqa: E402

SHARD = "/app/data/wp_shards"
N_SAMPLE = 400


def main() -> int:
    bids = open(f"{SHARD}/battle_ids.txt").read().splitlines()
    ids = np.load(f"{SHARD}/deck_ids.npy", mmap_mode="r")
    vars_ = np.load(f"{SHARD}/deck_vars.npy", mmap_mode="r")
    if len(ids) != len(bids):
        print(f"FAIL: deck_ids has {len(ids)} rows, battle_ids.txt has {len(bids)}")
        return 1

    session = get_session(get_engine(os.environ["DATABASE_URL"]))
    vocab = CardVocabulary(session)
    rng = random.Random(7)
    rows = sorted(rng.sample(range(len(bids)), min(N_SAMPLE, len(bids))))

    empty = mismatched = checked = 0
    for i in rows:
        bid = bids[i]
        db = session.execute(
            text("""SELECT is_player_deck, card_name, card_variant
                    FROM deck_cards WHERE battle_id=:b
                    ORDER BY is_player_deck DESC, card_name"""), {"b": bid}).all()
        if not db:
            empty += 1
            continue
        want = {0: [], 1: []}
        for is_own, name, variant in db:
            r = 0 if is_own == 1 else 1
            if len(want[r]) < DECK_SIZE:
                want[r].append((vocab.encode(name), VARIANT_IDX.get(variant or "base", 0)))
        checked += 1
        for r in (0, 1):
            got = sorted(zip(ids[i, r].tolist(), vars_[i, r].tolist()))
            exp = sorted(want[r] + [(0, 0)] * (DECK_SIZE - len(want[r])))
            if got != exp:
                mismatched += 1
                if mismatched <= 3:
                    print(f"MISMATCH {bid} side={r}\n  shard {got}\n  db    {exp}")

    nonzero = int((np.asarray(ids[rows]) != 0).sum())
    print(f"checked {checked} battles ({empty} had no deck rows), "
          f"{mismatched} side-mismatches, {nonzero} non-zero card slots")
    if mismatched:
        print("FAIL: shard decks disagree with the database")
        return 1
    if checked == 0 or nonzero == 0:
        print("FAIL: nothing verifiable — decks look empty")
        return 1
    print("PASS: shard deck arrays match the database")
    return 0


if __name__ == "__main__":
    sys.exit(main())
