"""Prove the incremental activity-profile merge equals a full rebuild.

The incremental path is only safe if folding a later slice into an earlier
aggregate yields exactly what aggregating everything at once would. That is the
whole claim, so it is tested rather than asserted: take a bounded set of
players, build their profiles in one pass, then build them as
(everything up to a split id) + (a top-up past it), and compare every field.

Bounded to a player sample so it runs in seconds against the primary instead of
adding another full-table scan to an instance we are trying to unload.

Run with cwd=/app:
  PYTHONPATH=/app/src python3 tools/diag/verify_activity_incremental.py [n_players]
"""

import os
import sys

from sqlalchemy import text

sys.path.insert(0, "/app/src")
from tracker.database import get_engine, get_session          # noqa: E402
from tracker.ml.activity_model import _merge_rows             # noqa: E402

AGG = """
    SELECT player_tag,
           EXTRACT(HOUR FROM battle_time) AS hour_utc,
           EXTRACT(ISODOW FROM battle_time) AS dow,
           COUNT(*) AS cnt,
           MIN(battle_time) AS first_bt,
           MAX(battle_time) AS last_bt
    FROM battles
    WHERE corpus IS NOT NULL AND battle_time IS NOT NULL
      AND player_tag IN :tags
      {idf}
    GROUP BY player_tag, hour_utc, dow
"""


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    session = get_session(get_engine(os.environ["DATABASE_URL"]))

    tags = tuple(r[0] for r in session.execute(text(
        "SELECT player_tag FROM battles WHERE corpus IS NOT NULL "
        "AND player_tag IS NOT NULL GROUP BY player_tag "
        "ORDER BY count(*) DESC LIMIT :n"), {"n": n}))
    if not tags:
        raise SystemExit("no corpus players found")

    # Split point: the median id among these players' battles, so both sides
    # are non-trivial (a split past the end would prove nothing).
    split = int(session.execute(text(
        "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY id) FROM battles "
        "WHERE corpus IS NOT NULL AND player_tag IN :tags"), {"tags": tags}).scalar())
    upto = int(session.execute(text(
        "SELECT MAX(id) FROM battles WHERE corpus IS NOT NULL AND player_tag IN :tags"),
        {"tags": tags}).scalar())

    full = {}
    _merge_rows(full, session.execute(text(AGG.format(idf="")), {"tags": tags}).all())

    inc = {}
    _merge_rows(inc, session.execute(
        text(AGG.format(idf="AND id <= :split")), {"tags": tags, "split": split}).all())
    before = sum(p["total_battles"] for p in inc.values())
    _merge_rows(inc, session.execute(
        text(AGG.format(idf="AND id > :split AND id <= :upto")),
        {"tags": tags, "split": split, "upto": upto}).all())

    print("players=%d  split_id=%d  max_id=%d" % (len(tags), split, upto))
    print("  battles before top-up: %d   after: %d   full: %d" % (
        before, sum(p["total_battles"] for p in inc.values()),
        sum(p["total_battles"] for p in full.values())))
    if before == sum(p["total_battles"] for p in inc.values()):
        print("  WARNING: top-up added nothing — the split proved nothing")

    bad = 0
    if set(full) != set(inc):
        print("  MISMATCH: player sets differ (%d vs %d)" % (len(full), len(inc)))
        bad += 1
    for tag in sorted(set(full) & set(inc)):
        a, b = full[tag], inc[tag]
        for k in ("hourly_counts", "dow_counts", "total_battles",
                  "first_battle_time", "last_battle_time"):
            if a[k] != b[k]:
                bad += 1
                if bad <= 5:
                    print("  MISMATCH %s %s: full=%r incremental=%r" % (tag, k, a[k], b[k]))

    print("VERDICT: %s (%d field mismatches)" % ("IDENTICAL" if bad == 0 else "BROKEN", bad))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
