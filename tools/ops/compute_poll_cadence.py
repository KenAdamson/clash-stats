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

SQL = """
WITH recent AS (
    SELECT b.player_tag, b.battle_time,
           row_number() OVER (PARTITION BY b.player_tag
                              ORDER BY b.battle_time DESC) AS rn
    FROM battles b
    WHERE b.corpus = 'top_ladder'
      AND b.battle_time > now() - (:lookback || ' days')::interval
), spans AS (
    SELECT player_tag,
           count(*) AS n,
           EXTRACT(epoch FROM max(battle_time) - min(battle_time))/86400.0 AS span_d
    FROM recent
    WHERE rn <= :cap
    GROUP BY player_tag
), rated AS (
    SELECT player_tag,
           n / span_d AS rate,
           least(greatest(:cap / (n / span_d), :min_cad), :max_cad) AS cadence
    FROM spans
    WHERE n >= :min_n AND span_d > :min_span
)
UPDATE player_corpus pc
SET play_rate = r.rate,
    poll_cadence_days = r.cadence,
    rate_measured_at = now()
FROM rated r
WHERE pc.player_tag = r.player_tag
  AND pc.active = 1
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    session = get_session(get_engine(os.environ["DATABASE_URL"]))
    params = {
        "lookback": str(LOOKBACK_DAYS), "cap": WINDOW_CAP,
        "min_n": MIN_BATTLES, "min_span": MIN_SPAN_DAYS,
        "min_cad": MIN_CADENCE_DAYS, "max_cad": MAX_CADENCE_DAYS,
    }
    logger.info("window_cap=%d lookback=%dd clamps=[%.2f, %.1f]d",
                WINDOW_CAP, LOOKBACK_DAYS, MIN_CADENCE_DAYS, MAX_CADENCE_DAYS)

    if args.dry:
        row = session.execute(text(SQL.replace(
            "UPDATE player_corpus pc\nSET play_rate = r.rate,\n"
            "    poll_cadence_days = r.cadence,\n    rate_measured_at = now()\n"
            "FROM rated r\nWHERE pc.player_tag = r.player_tag\n  AND pc.active = 1",
            "SELECT count(*) AS n, round(avg(rate)::numeric,2) AS mean_rate,\n"
            "       round(avg(cadence)::numeric,2) AS mean_cadence,\n"
            "       round(sum(1.0/cadence)) AS polls_per_day_required\n"
            "FROM rated")), params).one()
        logger.info("DRY: %d players measurable, mean rate %.2f/day, "
                    "mean cadence %.2fd, budget required %s polls/day",
                    row.n, row.mean_rate, row.mean_cadence,
                    row.polls_per_day_required)
        return

    t0 = time.time()
    res = session.execute(text(SQL), params)
    session.commit()
    logger.info("updated %d players in %.1f min", res.rowcount or 0,
                (time.time() - t0) / 60)

    row = session.execute(text("""
        SELECT count(*) FILTER (WHERE play_rate IS NOT NULL) AS rated,
               count(*) AS active_total,
               round(avg(play_rate)::numeric, 2) AS mean_rate,
               round(sum(1.0/poll_cadence_days)) AS polls_day_required
        FROM player_corpus WHERE active = 1
    """)).one()
    logger.info("active=%d rated=%d (%.1f%%) mean_rate=%.2f/day "
                "required=%s polls/day",
                row.active_total, row.rated,
                100.0 * row.rated / max(row.active_total, 1),
                row.mean_rate, row.polls_day_required)


if __name__ == "__main__":
    main()
