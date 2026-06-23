"""Deck-invariant pilot fingerprint — smurf-score pillar 3 (behavioral skill).

The fingerprint is a 6-dimensional behavioral vector extracted from a player's
replay placements (``replay_events``), engineered so it captures the PILOT and
not the deck:

  elixir_pace      median(gap_ticks / elixir_cost_of_card_just_played)  — banking
  throughput       own elixir spent / own-play span (per 1000 ticks)    — controlled spend
  reaction         median latency prior-opp-placement -> next own play (<=300t)
  pace_consistency stdev(gap/cost) / mean(gap/cost)  (CV; discipline = low)
  def_reaction     median latency opp-placement -> next own placement in OWN half (defense snap)
  fast_react_frac  fraction of reactions <= 50 ticks (snap vs deliberate)

Raw tempo and the spatial features (lane / aggression) are deliberately NOT
used: tempo scales with deck cycle cost, and lane/aggression encode deck ROLE
(an aggressive opener vs a reactive punish deck) rather than the pilot. See the
2026-06-22 validation: the user's main and alt — zero shared cards, ~9000
trophy gap — land at #2 of 405 nearest neighbors, at same-pilot distance, with
self-consistency AUC 0.83.

The match score uses plain z-Euclidean distance. We tested whitening /
Mahalanobis and it BACKFIRED (AUC 0.78, near-random cross-account rank): the
timing features are correlated *because* they measure one real trait, and
decorrelating throws that signal away. Do not whiten.

``behavioral_gap`` (written onto ``player_dim``) = median trophies of an
account's k nearest pilots minus its own trophies. A large positive gap means
the account *plays like* a much-higher-trophy pilot — the skill-smurf signal,
orthogonal to clan-shelter (pillar 1) and the funded/level gap (pillar 2).
"""
from __future__ import annotations

import logging
import os
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import bindparam, select, text
from sqlalchemy.orm import Session

from tracker.models import PilotFingerprint, PlayerDim

logger = logging.getLogger(__name__)

FEATURES = ["elixir_pace", "throughput", "reaction", "pace_consistency",
            "def_reaction", "fast_react_frac"]

# Arena midpoints (CR replay coordinate space). y > YMID = opponent half.
YMID = 15750
REACT_MAX = 300       # ticks; a reaction must follow the opp play within this
FAST_REACT = 50       # ticks; "snap" answer threshold
IPI_MAX = 1200        # ignore between-game / pause gaps

MIN_GAMES = 20        # games needed for a stable fingerprint
GAME_CAP = 180        # most-recent games used per player (bounds query cost)
_REFRESH_BATCH = 500
_REFRESH_MAX_AGE_DAYS = 30
_KNN = 10             # neighbors for the behavioral-gap median
_OUR_CORPUSES = ("personal", "alt")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def slug(name: str) -> str:
    """Bridge display card names (``Baby Dragon``) to replay slugs (``baby-dragon``)."""
    return name.lower().replace(" ", "-").replace(".", "")


def load_card_costs(session: Session) -> dict[str, int]:
    """Map replay card-name slug -> elixir cost, from ``deck_cards``."""
    rows = session.execute(
        text("SELECT DISTINCT card_name, card_elixir FROM deck_cards "
             "WHERE card_elixir IS NOT NULL")
    ).fetchall()
    return {slug(n): e for n, e in rows if e and e > 0}


def _pull_games(session: Session, tag: str, cap: int = GAME_CAP) -> list[list[tuple]]:
    """Return the player's most-recent capped games as lists of events.

    Each event is ``(side, game_tick, card_name, arena_y)``. Capped in SQL via a
    subquery so there is no unbounded sort over heavily-scraped players.
    """
    rows = session.execute(
        text(
            """
            SELECT e.battle_id, e.side, e.game_tick, e.card_name, e.arena_y
            FROM replay_events e
            WHERE e.battle_id IN (
                SELECT b.battle_id FROM battles b
                WHERE b.player_tag = :tag AND b.replay_fetched = 1
                ORDER BY b.battle_time DESC NULLS LAST
                LIMIT :cap)
            ORDER BY e.battle_id, e.game_tick, e.id
            """
        ),
        {"tag": tag, "cap": cap},
    ).fetchall()
    by_game: dict[str, list[tuple]] = defaultdict(list)
    for _bid, side, tick, cname, ay in rows:
        by_game[_bid].append((side, tick, cname, ay))
    return list(by_game.values())


def _prev_opp_tick(opp_ticks: list[int], t: int) -> Optional[int]:
    prev = None
    for ot in opp_ticks:
        if ot < t:
            prev = ot
        else:
            break
    return prev


def compute_fingerprint(games: list[list[tuple]], costs: dict[str, int]) -> Optional[dict]:
    """Compute the 6-feature fingerprint over a set of games, or None if thin."""
    paces, reacts, def_reacts = [], [], []
    tot_elixir, tot_span, used = 0.0, 0.0, 0
    for evs in games:
        own = sorted([(t, c, ay) for (s, t, c, ay) in evs if s == "team"],
                     key=lambda r: r[0])
        if len(own) < 2:
            continue
        used += 1
        for a, b in zip(own, own[1:]):
            d = b[0] - a[0]
            if 0 < d <= IPI_MAX:
                c = costs.get(a[1])
                if c:
                    paces.append(d / c)
        tot_elixir += sum(costs.get(c, 0) for (_t, c, _y) in own)
        span = own[-1][0] - own[0][0]
        if span > 0:
            tot_span += span
        opp_ticks = sorted(t for (s, t, c, ay) in evs if s == "opponent")
        for t, c, ay in own:
            prev = _prev_opp_tick(opp_ticks, t)
            if prev is not None and 0 < t - prev <= REACT_MAX:
                reacts.append(t - prev)
                if ay is not None and ay < YMID:   # answered into own half = defensive
                    def_reacts.append(t - prev)
    if used < 8 or len(paces) < 20 or len(reacts) < 10 or tot_span <= 0:
        return None
    mean_pace = statistics.mean(paces)
    return {
        "elixir_pace": statistics.median(paces),
        "throughput": tot_elixir / tot_span * 1000,
        "reaction": float(statistics.median(reacts)),
        "pace_consistency": (statistics.pstdev(paces) / mean_pace) if mean_pace else 0.0,
        "def_reaction": float(statistics.median(def_reacts) if len(def_reacts) >= 5
                              else statistics.median(reacts)),
        "fast_react_frac": sum(1 for r in reacts if r <= FAST_REACT) / len(reacts),
        "n_games": used,
    }


def _latest_trophies(session: Session, tag: str) -> Optional[int]:
    return session.execute(
        text("SELECT MAX(player_starting_trophies) FROM battles WHERE player_tag = :tag"),
        {"tag": tag},
    ).scalar()


def _candidate_tags(session: Session) -> list[str]:
    """Players with >= MIN_GAMES replay'd team-side games."""
    rows = session.execute(
        text(
            """
            SELECT b.player_tag
            FROM replay_events e JOIN battles b ON b.battle_id = e.battle_id
            WHERE e.side = 'team'
            GROUP BY b.player_tag
            HAVING COUNT(DISTINCT e.battle_id) >= :mg
            """
        ),
        {"mg": MIN_GAMES},
    ).fetchall()
    return [r[0] for r in rows]


def _our_tags(session: Session) -> set[str]:
    rows = session.execute(
        text("SELECT DISTINCT player_tag FROM battles WHERE corpus IN :cs").bindparams(
            bindparam("cs", list(_OUR_CORPUSES), expanding=True)
        )
    ).fetchall()
    return {r[0] for r in rows}


def _opponent_tags(session: Session) -> set[str]:
    """Tags we actually face (player_dim) — pillar 3 lands on these, so they
    jump the fingerprint queue ahead of the rest of the corpus."""
    rows = session.execute(select(PlayerDim.player_tag)).fetchall()
    return {r[0] for r in rows}


def refresh_pilot_fingerprints(
    session: Session, batch: int = _REFRESH_BATCH,
    max_age_days: int = _REFRESH_MAX_AGE_DAYS,
) -> int:
    """Incrementally (re)compute fingerprints for up to ``batch`` pilots.

    Missing-then-stale, OUR accounts first — the same ours-first / chip-the-
    backlog pattern as the clan resolver, so a daily cron grows the reference
    pool without recomputing the whole corpus each run.

    Returns:
        Number of fingerprints written this run.
    """
    costs = load_card_costs(session)
    if not costs:
        logger.warning("pilot_fingerprint: no card costs available; skipping.")
        return 0

    candidates = set(_candidate_tags(session))
    if not candidates:
        return 0
    ours = _our_tags(session) & candidates
    opponents = _opponent_tags(session) & candidates
    existing = {t: ra for t, ra in session.execute(
        select(PilotFingerprint.player_tag, PilotFingerprint.refreshed_at)
    )}
    cutoff = _now() - timedelta(days=max_age_days)

    def is_stale(tag: str) -> bool:
        if tag not in existing:
            return True
        ra = existing[tag]
        if ra is None:
            return True
        if ra.tzinfo is None:
            ra = ra.replace(tzinfo=timezone.utc)
        return ra < cutoff

    def priority(tag: str) -> int:
        # ours -> opponents we face (pillar 3 targets) -> the rest of the corpus
        return 0 if tag in ours else 1 if tag in opponents else 2

    todo = [t for t in candidates if is_stale(t)]
    # priority tier, then missing-before-stale within a tier
    todo.sort(key=lambda t: (priority(t), t in existing))
    todo = todo[:batch]

    written = 0
    for tag in todo:
        try:
            fp = compute_fingerprint(_pull_games(session, tag), costs)
        except Exception as exc:  # never let one bad player abort the batch
            logger.warning("pilot_fingerprint: %s failed (%s)", tag, exc)
            session.rollback()
            continue
        if fp is None:
            continue
        row = session.get(PilotFingerprint, tag) or PilotFingerprint(player_tag=tag)
        for k in FEATURES:
            setattr(row, k, fp[k])
        row.n_games = fp["n_games"]
        row.latest_trophies = _latest_trophies(session, tag)
        row.refreshed_at = _now()
        session.merge(row)
        written += 1
    session.commit()
    logger.info("refresh_pilot_fingerprints: wrote %d (todo=%d, pool=%d)",
                written, len(todo), len(candidates))
    return written


def _load_matrix(session: Session):
    """Load all complete fingerprints -> (tags, Z, trophies, mean, std, raw)."""
    import numpy as np

    rows = session.execute(
        select(PilotFingerprint.player_tag, *[getattr(PilotFingerprint, k) for k in FEATURES],
               PilotFingerprint.latest_trophies)
        .where(PilotFingerprint.elixir_pace.isnot(None))
    ).fetchall()
    rows = [r for r in rows if all(r[i + 1] is not None for i in range(len(FEATURES)))]
    if not rows:
        return None
    tags = [r[0] for r in rows]
    raw = np.array([[r[i + 1] for i in range(len(FEATURES))] for r in rows], dtype=float)
    trophies = [r[-1] for r in rows]
    mean = raw.mean(0)
    std = raw.std(0)
    std[std == 0] = 1.0
    Z = (raw - mean) / std
    return tags, Z, trophies, mean, std, raw


def compute_behavioral_match(session: Session, k: int = _KNN) -> int:
    """Write ``behavioral_neighbor_trophy`` / ``behavioral_gap`` onto player_dim.

    For each player_dim row that has a fingerprint, find its ``k`` nearest pilots
    (z-Euclidean, excluding self), take the median of their trophies, and store
    that and the gap to the account's own trophies. Returns rows updated.
    """
    import numpy as np

    loaded = _load_matrix(session)
    if loaded is None:
        return 0
    tags, Z, trophies, _mean, _std, _raw = loaded
    idx = {t: i for i, t in enumerate(tags)}

    pd_rows = {r.player_tag: r for r in session.query(PlayerDim).all()}
    updated = 0
    for tag, pd in pd_rows.items():
        i = idx.get(tag)
        if i is None:
            continue
        d = np.linalg.norm(Z - Z[i], axis=1)
        order = np.argsort(d)
        neigh_troph = []
        for j in order:
            if j == i:
                continue
            if trophies[j] is not None:
                neigh_troph.append(trophies[j])
            if len(neigh_troph) >= k:
                break
        if not neigh_troph:
            continue
        nt = int(statistics.median(neigh_troph))
        base = pd.latest_trophies if pd.latest_trophies is not None else trophies[i]
        pd.behavioral_neighbor_trophy = nt
        pd.behavioral_gap = (nt - base) if base is not None else None
        updated += 1
    session.commit()
    logger.info("compute_behavioral_match: updated %d player_dim rows", updated)
    return updated


def nearest_pilots(session: Session, tag: str, k: int = 15) -> list[dict]:
    """Rank the pilots whose fingerprint is closest to ``tag`` (CLI/analysis)."""
    import numpy as np

    loaded = _load_matrix(session)
    if loaded is None:
        return []
    tags, Z, trophies, _mean, _std, _raw = loaded
    idx = {t: i for i, t in enumerate(tags)}
    if tag not in idx:
        return []
    i = idx[tag]
    d = np.linalg.norm(Z - Z[i], axis=1)
    order = [j for j in np.argsort(d) if j != i][:k]
    return [{"player_tag": tags[j], "distance": float(d[j]),
             "trophies": trophies[j]} for j in order]
