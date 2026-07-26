"""Analytic per-tick elixir reconstruction from the placement stream.

Clash Royale elixir is a closed-form function of time (regen) minus spend
(card costs at their placement ticks), capped at 10. The overflow at the cap
IS the "elixir leaked" signal RoyaleAPI reports — so computing the trace both
yields exact elixir-in-hand at every tick AND regenerates leak as a
by-product, validating the physics against ground truth.

The game gives NO board-state labels, but this subsystem is simple enough to
solve exactly from placements alone — the first genuine slice of board-truth
with no game engine and no vision (see the WP board-state discussion).

Elixir schedule (elapsed seconds):
  0:00-2:00  1x  1 elixir / 2.8s   (first 2 min of regular)
  2:00-3:00  2x  1 / 1.4s          (last min of regular, "2x ELIXIR")
  3:00+      3x  1 / 0.933s        (overtime)
Bar caps at 10; overflow accrues as leak.

Tick->seconds: 20 ticks/sec (50ms engine tick). VERIFIED 2026-07-09 by fitting
computed leak to RoyaleAPI ground-truth elixirLeaked (20Hz best, monotonically
better than 10/12/15/18.7) AND by in-game clock anchors (Witch tick 214 @ ~11s,
game end tick 1956 @ ~104s-6s-fall). NOTE: temporal_analysis.py's 18.667
(3360t==3:00) is a ~7% mis-calibration; at 20Hz, 3:00 == 3600 ticks.
"""
from __future__ import annotations

TICKS_PER_SEC = 20.0  # 50ms engine tick — verified vs ground-truth leak + clock anchors
ELIXIR_CAP = 10.0

# (phase_end_seconds, elixir_per_second)
_SCHEDULE = [
    (120.0, 1.0 / 2.8),    # 1x
    (180.0, 1.0 / 1.4),    # 2x
    (float("inf"), 1.0 / 0.933),  # 3x overtime
]


def _regen(t0: float, t1: float) -> float:
    """Elixir regenerated between elapsed seconds t0 and t1 (piecewise rate)."""
    total = 0.0
    lo = t0
    for edge, rate in _SCHEDULE:
        if lo >= t1:
            break
        hi = min(edge, t1)
        if hi > lo:
            total += (hi - lo) * rate
            lo = hi
    return total


def elixir_trace(events, start_elixir: float = 5.0, ticks_per_sec: float = TICKS_PER_SEC,
                 accrue_to_end: bool = True):
    """Per-side elixir-in-hand and cumulative leak at each of that side's plays.

    Args:
        events: iterable of (side, game_tick, elixir_cost) in tick order; side
            is "team" or "opponent". elixir_cost is the card's cost.
        start_elixir: elixir in hand at tick 0 (bar pre-fill at match start).

    Returns:
        dict side -> list of (tick, elixir_before_play, elixir_after_play,
        cumulative_leak). Plus each side's final leak accrued to the last
        observed tick.
    """
    state = {"team": [start_elixir, 0.0, 0.0], "opponent": [start_elixir, 0.0, 0.0]}
    # [elixir, last_time_s, cum_leak]
    out = {"team": [], "opponent": []}
    max_tick = 0
    for side, tick, cost in events:
        if side not in state:
            continue
        max_tick = max(max_tick, tick)
        st = state[side]
        t_now = tick / ticks_per_sec
        gained = _regen(st[1], t_now)
        raw = st[0] + gained
        if raw > ELIXIR_CAP:
            st[2] += raw - ELIXIR_CAP  # leak = overflow
            raw = ELIXIR_CAP
        before = raw
        after = max(0.0, raw - (cost or 0))
        st[0] = after
        st[1] = t_now
        out[side].append((tick, before, after, st[2]))
    # Accrue leak from each side's last play to game end. ON by default:
    # tested 2026-07-09 — RoyaleAPI's elixirLeaked DOES count the winner
    # idling at 10 post-decision (removing it hurt correlation 0.54->0.47 and
    # didn't fix the +2 bias). The residual overprediction is elsewhere —
    # most likely occasional placements missing from the replay parse
    # (un-subtracted spend -> apparent over-hoard). corr~0.54 caps validation.
    final_leak = {}
    if accrue_to_end:
        t_end = max_tick / ticks_per_sec
        for side, st in state.items():
            raw = st[0] + _regen(st[1], t_end)
            if raw > ELIXIR_CAP:
                st[2] += raw - ELIXIR_CAP
    for side, st in state.items():
        final_leak[side] = st[2]
    return out, final_leak


def per_event_elixir(events, start_elixir: float = 5.0,
                     ticks_per_sec: float = TICKS_PER_SEC):
    """Elixir-in-hand for both sides at each event, aligned to input order.

    For every event, returns (own_before, opp_now, diff) where own_before is
    the acting side's elixir just before the play, opp_now is the other side's
    elixir at that same tick (regen since their last play, capped), and diff =
    own_before - opp_now (positive = the actor holds an elixir advantage). This
    is exact board-truth derived purely from placements — the running elixir
    economy the WP model's game-level aggregates can't express.

    Returns a list aligned 1:1 with ``events``.
    """
    st = {"team": [start_elixir, 0.0], "opponent": [start_elixir, 0.0]}  # [elixir, last_time_s]

    def _elixir_at(side, t_s):
        e = st[side]
        return min(ELIXIR_CAP, e[0] + _regen(e[1], t_s))

    result = []
    for side, tick, cost in events:
        if side not in st:
            result.append((0.0, 0.0, 0.0))
            continue
        other = "opponent" if side == "team" else "team"
        t_s = tick / ticks_per_sec
        own_before = _elixir_at(side, t_s)
        opp_now = _elixir_at(other, t_s)
        result.append((own_before, opp_now, own_before - opp_now))
        # commit the acting side's spend; other side's balance is read-only here
        st[side][0] = max(0.0, own_before - (cost or 0))
        st[side][1] = t_s
    return result
