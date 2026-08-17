"""Check the corrected aggression/lane features against ground truth.

The polarity fix is only believable if the corrected feature behaves like
aggression should. Two independent checks, on real games:

1. ROTATION SYMMETRY. Lane choice is strategically symmetric (left and right
   towers are interchangeable), so corpus-wide the corrected "right lane" rate
   must sit near 0.5 in BOTH rotated and non-rotated games. If the correction
   were wrong, the two cohorts would diverge -- that divergence is exactly the
   bug's fingerprint.

2. COHORT CONSISTENCY (the decisive one). Rotation is an artefact of how the
   replay was fetched, not of how anyone played, so the two cohorts are
   statistically the same population and any real feature must read the same in
   both. The old definition scores the SAME behaviour ~0.79 in non-rotated games
   and ~0.15 in rotated ones -- it was aggression in half the corpus and defence
   in the other half, a mixture of a quantity and its own complement.

3. OUTCOME DIRECTION. Placing in the opponent's half is offence, so aggression
   should be higher in won games. Note this does NOT cleanly invert for the old
   feature: a 50/50 mixture of opposite meanings blends the two deltas rather
   than flipping the sign, so it is corroboration, not proof.

Run with cwd=/app:
  PYTHONPATH=/app/src python3 tools/diag/verify_features_orientation.py [n_games]
"""

import os
import sys
from collections import defaultdict

from sqlalchemy import text

sys.path.insert(0, "/app/src")
from tracker.database import get_engine, get_session      # noqa: E402
from tracker.ml.features import (                         # noqa: E402
    ARENA_X_MID, ARENA_Y_MID, _is_rotated, _oriented_xy,
)


class _Ev:
    __slots__ = ("arena_x", "arena_y", "side")

    def __init__(self, x, y, side):
        self.arena_x, self.arena_y, self.side = x, y, side


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    session = get_session(get_engine(os.environ["DATABASE_URL"]))

    rows = session.execute(text("""
        SELECT re.battle_id, re.side, re.arena_x, re.arena_y, b.result
        FROM replay_events re
        JOIN battles b ON b.battle_id = re.battle_id
        WHERE re.arena_x IS NOT NULL AND re.arena_y IS NOT NULL
          AND b.result IN ('win','loss')
          AND re.battle_id IN (
              SELECT battle_id FROM game_wp_summary ORDER BY battle_id LIMIT :n)
    """), {"n": n}).all()

    games = defaultdict(lambda: {"team": [], "opponent": [], "result": None})
    for bid, side, x, y, res in rows:
        g = games[bid]
        g["result"] = res
        if side in ("team", "opponent"):
            g[side].append(_Ev(x, y, side))

    stats = defaultdict(lambda: {"agg": [], "old_agg": [], "right": []})
    for bid, g in games.items():
        team, opp = g["team"], g["opponent"]
        if len(team) < 3 or len(opp) < 3:
            continue
        rot = _is_rotated(team, opp)
        agg = sum(1 for e in team if _oriented_xy(e, rot)[1] < ARENA_Y_MID) / len(team)
        old = sum(1 for e in team if e.arena_y > ARENA_Y_MID) / len(team)
        right = sum(1 for e in team if _oriented_xy(e, rot)[0] > ARENA_X_MID) / len(team)
        for key in (("rot" if rot else "norm"), g["result"], "ALL"):
            stats[key]["agg"].append(agg)
            stats[key]["old_agg"].append(old)
            stats[key]["right"].append(right)

    def m(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    print("cohort      games   aggression(fixed)  aggression(old)  right-lane(fixed)")
    for key in ("norm", "rot", "win", "loss", "ALL"):
        s = stats[key]
        print("%-10s %6d   %14.3f   %14.3f   %15.3f" % (
            key, len(s["agg"]), m(s["agg"]), m(s["old_agg"]), m(s["right"])))

    lane_gap = abs(m(stats["norm"]["right"]) - m(stats["rot"]["right"]))
    fixed_delta = m(stats["win"]["agg"]) - m(stats["loss"]["agg"])
    old_delta = m(stats["win"]["old_agg"]) - m(stats["loss"]["old_agg"])

    old_gap = abs(m(stats["norm"]["old_agg"]) - m(stats["rot"]["old_agg"]))
    new_gap = abs(m(stats["norm"]["agg"]) - m(stats["rot"]["agg"]))

    print("\n1. rotation symmetry  : right-lane gap rot vs norm = %.3f %s"
          % (lane_gap, "OK (<0.05)" if lane_gap < 0.05 else "SUSPECT"))
    print("2. cohort consistency : aggression gap  old %.3f -> fixed %.3f  (%.1fx better) %s"
          % (old_gap, new_gap, old_gap / new_gap if new_gap else float("inf"),
             "PASS" if new_gap < old_gap / 3 else "FAIL"))
    print("3. outcome direction  : win-loss aggression, fixed %+.4f %s"
          % (fixed_delta, "PASS (wins press)" if fixed_delta > 0 else "FAIL"))
    print("   (old %+.4f -- blended, not a clean inversion; see docstring)" % old_delta)


if __name__ == "__main__":
    main()
