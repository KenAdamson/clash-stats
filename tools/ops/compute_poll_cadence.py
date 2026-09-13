"""Measure each corpus player's play rate and derive a poll cadence.

WHY A RATE AND NOT AN ACTIVITY SCORE
The existing activity model answers "will this player have new battles?", which
after a ~10-day sweep is almost always yes -- it cannot discriminate in the
regime we actually operate in (AUC 0.971 on a balanced train set, but the live
question is near-degenerate). The operational question is not IF but HOW MANY:
a poll returns at most a full ~30-battle window, so what matters is when a
player fills it.

WHY THE SPAN TRICK WORKS
rate = n / (max(battle_time) - min(battle_time)) over the most recent captured
battles. This is poll-independent and remains EXACT under truncation: 30
battles spanning six hours is 120 games/day whether or not we missed earlier
ones. That is what makes this measurable from data already on disk instead of
needing new instrumentation and weeks of accumulation.

WHAT IT IS NOT
It is a point estimate of recent behaviour, not a forecast. It cannot see
time-of-day or weekday structure, and it will lag a season-start surge. Those
are exactly the gaps a learned model should fill later; this deliberately ships
first because it is deterministic, explainable, and cannot silently regress.

Run with cwd=/app:
  PYTHONPATH=/app/src python3 tools/ops/compute_poll_cadence.py [--dry]
"""

import argparse
import logging
import os
import sys
import time

from sqlalchemy import text

sys.path.insert(0, "/app/src")
from tracker.database import get_engine, get_session   # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("poll_cadence")

# The CR battlelog window. Logged polls top out here, so it is the point past
# which additional games are lost rather than merely delayed.
WINDOW_CAP = int(os.environ.get("CR_BATTLE_WINDOW", "30"))
# Below these the span estimate is noise: a handful of battles inside a few
# minutes implies an absurd rate, and one long-idle pair implies a fake-low one.
MIN_BATTLES = 5
MIN_SPAN_DAYS = 0.04          # ~1 hour
LOOKBACK_DAYS = int(os.environ.get("CADENCE_LOOKBACK_DAYS", "60"))
# Clamps. The floor stops a 100-games/day player demanding 3 polls a day and
# eating the budget; the ceiling stops a barely-active player drifting to a
# cadence so long we would never revisit them at all.
MIN_CADENCE_DAYS = float(os.environ.get("CADENCE_MIN_DAYS", "0.5"))
MAX_CADENCE_DAYS = float(os.environ.get("CADENCE_MAX_DAYS", "21"))

# Per-player LATERAL probes, NOT a global window function.
#
# The obvious form -- row_number() OVER (PARTITION BY player_tag ORDER BY
# battle_time DESC) across the whole corpus -- has to sort every corpus battle
# in the lookback window (millions of rows) before it can take the top 30 of
# each. That ran 11+ minutes and drove the box into memory pressure hard enough
# to get the client killed, leaving the backend orphaned and still running.
#
# LATERAL ... ORDER BY battle_time DESC LIMIT :cap instead walks
# idx_battles_corpus_player_time once per player and stops after 30 rows. Same
# answer, bounded memory. This is the same rewrite that took an unrelated query
# here from 68s to 0.6s; the pattern is worth reaching for by default.
#
# Batched by player so no single transaction holds a long write lock on
# player_corpus while the every-minute scrape is trying to update last_scraped.
SQL = """
WITH batch AS (
    SELECT player_tag FROM player_corpus
    WHERE active = 1 AND player_tag > :after
    ORDER BY player_tag LIMIT :batch
), rated AS (
    SELECT b.player_tag,
           s.n / s.span_d AS rate,
           least(greatest(:cap / (s.n / s.span_d), :min_cad), :max_cad) AS cadence
    FROM batch b
    CROSS JOIN LATERAL (
        SELECT count(*) AS n,
               EXTRACT(epoch FROM max(bt) - min(bt))/86400.0 AS span_d
        FROM (
            SELECT bb.battle_time AS bt
            FROM battles bb
            WHERE bb.corpus = 'top_ladder'
              AND bb.player_tag = b.player_tag
              AND bb.battle_time > now() - (:lookback || ' days')::interval
            ORDER BY bb.battle_time DESC
            LIMIT :cap
        ) x
    ) s
    WHERE s.n >= :min_n AND s.span_d > :min_span
)
UPDATE player_corpus pc
SET play_rate = r.rate,
    poll_cadence_days = r.cadence,
    rate_measured_at = now()
FROM rated r
WHERE pc.player_tag = r.player_tag
"""

MAX_TAG = "\uffff"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=5000)
    ap.add_argument("--dry", action="store_true",
                    help="measure one batch and report, writing nothing")
    args = ap.parse_args()

    session = get_session(get_engine(os.environ["DATABASE_URL"]))
    base = {
        "lookback": str(LOOKBACK_DAYS), "cap": WINDOW_CAP,
        "min_n": MIN_BATTLES, "min_span": MIN_SPAN_DAYS,
        "min_cad": MIN_CADENCE_DAYS, "max_cad": MAX_CADENCE_DAYS,
        "batch": args.batch,
    }
    logger.info("window_cap=%d lookback=%dd clamps=[%.2f, %.1f]d batch=%d",
                WINDOW_CAP, LOOKBACK_DAYS, MIN_CADENCE_DAYS, MAX_CADENCE_DAYS,
                args.batch)

    if args.dry:
        # Same measurement, one batch, no write -- so the plan and the runtime
        # can be sanity-checked before committing to the full sweep.
        probe = SQL[SQL.index("WITH batch"):SQL.index("UPDATE player_corpus")]
        row = session.execute(
            text(probe + "SELECT count(*) AS n, round(avg(rate)::numeric,2) AS mean_rate, "
                         "round(avg(cadence)::numeric,2) AS mean_cad FROM rated"),
            dict(base, after="")).one()
        logger.info("DRY on %d players: %d measurable, mean rate %.2f/day, "
                    "mean cadence %.2fd", args.batch, row.n or 0,
                    row.mean_rate or 0, row.mean_cad or 0)
        return

    after = ""
    total = 0
    t0 = time.time()
    while True:
        # Advance the cursor by the BATCH's last tag, not the updated rows':
        # players filtered out for too few battles are never updated, and
        # keying on updated rows would loop forever on the first such gap.
        last = session.execute(
            text("SELECT player_tag FROM player_corpus WHERE active = 1 "
                 "AND player_tag > :after ORDER BY player_tag "
                 "OFFSET :off LIMIT 1"),
            {"after": after, "off": args.batch - 1}).scalar()

        res = session.execute(text(SQL), dict(base, after=after))
        session.commit()
        total += res.rowcount or 0

        if last is None:
            break
        after = last
        if (total // args.batch) % 5 == 0:
            logger.info("  %d rated so far (%.1f min)", total,
                        (time.time() - t0) / 60)

    logger.info("updated %d players in %.1f min", total, (time.time() - t0) / 60)

    row = session.execute(text("""
        SELECT count(*) FILTER (WHERE play_rate IS NOT NULL) AS rated,
               count(*) AS active_total,
               round(avg(play_rate)::numeric, 2) AS mean_rate,
               round(percentile_cont(0.5) WITHIN GROUP (ORDER BY play_rate)::numeric,2) AS median_rate,
               round(sum(1.0/poll_cadence_days)) AS polls_day_required
        FROM player_corpus WHERE active = 1
    """)).one()
    logger.info("active=%d rated=%d (%.1f%%) mean=%.2f/day median=%.2f/day "
                "required=%s polls/day",
                row.active_total, row.rated,
                100.0 * row.rated / max(row.active_total, 1),
                row.mean_rate or 0, row.median_rate or 0,
                row.polls_day_required)


if __name__ == "__main__":
    main()
